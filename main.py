import pandas as pd
import re
import numpy as np
from typing import List, Dict, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
import os
import gc
import time
from dataclasses import dataclass
import json

@dataclass
class ChunkingStrategy:
    name: str
    function: callable
    params: Dict
    param_ranges: Dict

class ChunkOptimizer:
    def __init__(self, docs, questions):
        self.docs = docs
        self.questions = questions
        self.model = None
        self.results = []
        
    def initialize_model(self):
        """Инициализация модели эмбеддингов"""
        if self.model is None:
            print("Загрузка модели эмбеддингов...")
            self.model = SentenceTransformer("ai-forever/ru-en-RoSBERTa")
    
    def clear_memory(self):
        """Очистка памяти"""
        gc.collect()
    
    def split_sentences(self, text: str, min_sentence_length: int = 10, **kwargs) -> List[str]:
        """Разбиение по предложениям с минимальной длиной"""
        if not text or len(text.strip()) == 0:
            return []
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) 
                    if len(s.strip()) > min_sentence_length]
        return sentences
    
    def split_paragraphs(self, text: str, min_paragraph_length: int = 30, **kwargs) -> List[str]:
        """Разбиение по абзацам с минимальной длиной"""
        if not text or len(text.strip()) == 0:
            return []
        paragraphs = [p.strip() for p in text.split("\n\n") 
                     if len(p.strip()) > min_paragraph_length]
        return paragraphs
    
    def split_fixed(self, text: str, chunk_size: int = 100, **kwargs) -> List[str]:
        """Разбиение на куски фиксированной длины"""
        if not text or len(text.strip()) == 0:
            return []
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > kwargs.get('min_chunk_length', 20):
                chunks.append(chunk)
        return chunks
    
    def split_overlap(self, text: str, chunk_size: int = 100, overlap: int = 20, **kwargs) -> List[str]:
        """Разбиение с перекрытием"""
        if not text or len(text.strip()) == 0:
            return []
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            if len(chunk.strip()) > kwargs.get('min_chunk_length', 20):
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks
    
    def split_headings(self, text: str, min_chunk_length: int = 50, **kwargs) -> List[str]:
        """Разбиение по заголовкам"""
        if not text or len(text.strip()) == 0:
            return []
        
        # Ищем заголовки h1., h2., h3.
        heading_pattern = r'(^h[1-3]\.\s+.*?)(?=\n\n|$|\nh[1-3]\.\s+)'
        chunks = []
        
        matches = list(re.finditer(heading_pattern, text, re.DOTALL | re.MULTILINE))
        
        if not matches:
            return self.split_paragraphs(text, min_paragraph_length=min_chunk_length)
        
        for i, match in enumerate(matches):
            start_pos = match.start()
            
            if i < len(matches) - 1:
                end_pos = matches[i + 1].start()
                chunk_content = text[start_pos:end_pos].strip()
            else:
                chunk_content = text[start_pos:].strip()
            
            if len(chunk_content) > min_chunk_length:
                chunks.append(chunk_content)
        
        return chunks
    
    def create_chunks_dataset(self, strategy_func: callable, strategy_params: Dict) -> pd.DataFrame:
        """Создает датасет из разбитых документов с заданными параметрами"""
        chunks_data = []
        
        for index, row in self.docs.iterrows():
            text = row["text"]
            page_id = row["page_id"]
            
            if not text or len(text.strip()) == 0:
                continue
                
            chunks = strategy_func(text, **strategy_params)
            
            for chunk_idx, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) > strategy_params.get('min_chunk_length', 30):
                    chunks_data.append({
                        "text": chunk_text,
                        "page_id": page_id,
                        "chunk_id": f"{page_id}_{chunk_idx}",
                        "words_count": len(chunk_text.split())
                    })
        
        return pd.DataFrame(chunks_data)
    
    def evaluate_strategy(self, chunks_df: pd.DataFrame, strategy_name: str, params: Dict) -> Dict:
        """Оценка стратегии с заданными параметрами"""
        self.clear_memory()
        
        # Создаем уникальное имя для коллекции
        collection_name = f"{strategy_name}_{hash(str(params))}"
        
        # Создаем векторную базу
        client = chromadb.PersistentClient(f"./data/chroma_optimization")
        
        try:
            client.delete_collection(collection_name)
        except:
            pass
        
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Добавляем chunks в векторную базу
        batch_size = 30
        total_chunks = len(chunks_df)
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks_df.iloc[i:i + batch_size]
            
            texts = batch["text"].tolist()
            embeddings = self.model.encode(texts, show_progress_bar=False)
            metadatas = [{"page_id": row["page_id"], "chunk_id": row["chunk_id"]} 
                        for _, row in batch.iterrows()]
            ids = [f"id_{i+j}" for j in range(len(batch))]
            
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )
        
        # Оценка качества
        df_data = []
        
        for _, row in self.questions.iterrows():
            question = row["question"]
            target_page_id = row["page_id"]
            query = self.model.encode(question)
            
            results = collection.query(
                query_embeddings=[query.tolist()], 
                n_results=5
            )
            
            found = False
            for n, (meta, score) in enumerate(
                zip(results["metadatas"][0], results["distances"][0]),
                start=1
            ):
                retrieved_page_id = meta["page_id"]
                if str(target_page_id) == str(retrieved_page_id):
                    data_row = [question, n, score]
                    df_data.append(data_row)
                    found = True
                    break
            
            if not found:
                data_row = [question, 0, 0]
                df_data.append(data_row)
        
        # Расчет метрик
        df = pd.DataFrame(df_data, columns=["question", "position", "score"])
        df["rank"] = df["position"].apply(lambda x: 1 / x if x != 0 else 0)
        MRR = df["rank"].sum() / len(self.questions)
        
        hits_at_1 = len(df[df["position"] == 1])
        hits_at_3 = len(df[df["position"] <= 3])
        hits_at_5 = len(df[df["position"] <= 5])
        
        # Статистика по chunks
        avg_words = chunks_df['words_count'].mean() if len(chunks_df) > 0 else 0
        total_chunks = len(chunks_df)
        
        # Очистка
        try:
            client.delete_collection(collection_name)
        except:
            pass
        
        self.clear_memory()
        
        return {
            "strategy": strategy_name,
            "params": params.copy(),
            "MRR": MRR,
            "Hits@1": hits_at_1 / len(self.questions),
            "Hits@3": hits_at_3 / len(self.questions),
            "Hits@5": hits_at_5 / len(self.questions),
            "avg_chunk_words": avg_words,
            "total_chunks": total_chunks,
            "chunk_efficiency": MRR / max(1, total_chunks / len(self.docs))  # MRR на chunk
        }
    
    def generate_parameter_combinations(self, strategy_name: str) -> List[Dict]:
        """Генерация комбинаций параметров для стратегии"""
        param_combinations = []
        
        if strategy_name == "sentences":
            for min_length in [5, 10, 15, 20]:
                param_combinations.append({
                    'min_sentence_length': min_length,
                    'min_chunk_length': min_length * 3
                })
                
        elif strategy_name == "paragraphs":
            for min_length in [20, 30, 50, 80]:
                param_combinations.append({
                    'min_paragraph_length': min_length,
                    'min_chunk_length': min_length
                })
                
        elif strategy_name == "fixed":
            for chunk_size in [50, 100, 150, 200, 250]:
                param_combinations.append({
                    'chunk_size': chunk_size,
                    'min_chunk_length': 20
                })
                
        elif strategy_name == "overlap":
            for chunk_size in [100, 150, 200]:
                for overlap in [10, 20, 30, 40]:
                    if overlap < chunk_size:
                        param_combinations.append({
                            'chunk_size': chunk_size,
                            'overlap': overlap,
                            'min_chunk_length': 20
                        })
                        
        elif strategy_name == "headings":
            for min_length in [30, 50, 80, 100]:
                param_combinations.append({
                    'min_chunk_length': min_length
                })
        
        return param_combinations
    
    def optimize_strategy(self, strategy_name: str, max_experiments: int = 10):
        """Оптимизация параметров для конкретной стратегии"""
        print(f"\n🔧 Оптимизация стратегии: {strategy_name}")
        print("=" * 50)
        
        self.initialize_model()
        
        # Получаем функцию стратегии
        strategy_func = getattr(self, f"split_{strategy_name}")
        
        # Генерируем комбинации параметров
        param_combinations = self.generate_parameter_combinations(strategy_name)
        
        if len(param_combinations) > max_experiments:
            # Выбираем наиболее разнообразные комбинации
            param_combinations = param_combinations[:max_experiments]
        
        strategy_results = []
        
        for i, params in enumerate(param_combinations):
            print(f"Эксперимент {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Создаем chunks
                chunks_df = self.create_chunks_dataset(strategy_func, params)
                
                if len(chunks_df) == 0:
                    print("  ⚠️  Не создано chunks, пропускаем")
                    continue
                
                # Оцениваем стратегию
                result = self.evaluate_strategy(chunks_df, strategy_name, params)
                strategy_results.append(result)
                
                print(f"  ✅ MRR: {result['MRR']:.3f}, Chunks: {result['total_chunks']}, "
                      f"Avg words: {result['avg_chunk_words']:.1f}")
                
            except Exception as e:
                print(f"  ❌ Ошибка: {e}")
                continue
        
        return strategy_results
    
    def run_comprehensive_optimization(self):
        """Запуск комплексной оптимизации всех стратегий"""
        print("🚀 ЗАПУСК КОМПЛЕКСНОЙ ОПТИМИЗАЦИИ ЧАНКОВ")
        print("=" * 60)
        
        strategies = ["sentences", "paragraphs", "fixed", "overlap", "headings"]
        all_results = []
        
        for strategy in strategies:
            strategy_results = self.optimize_strategy(strategy)
            all_results.extend(strategy_results)
            
            # Находим лучший результат для этой стратегии
            if strategy_results:
                best_for_strategy = max(strategy_results, key=lambda x: x['MRR'])
                print(f"🏆 Лучший результат для {strategy}: MRR = {best_for_strategy['MRR']:.3f}")
        
        # Сохраняем все результаты
        self.save_results(all_results)
        
        # Анализируем результаты
        self.analyze_results(all_results)
        
        return all_results
    
    def save_results(self, results: List[Dict]):
        """Сохранение результатов оптимизации"""
        os.makedirs("./data/optimization_results", exist_ok=True)
        
        # Сохраняем в CSV
        df_results = pd.DataFrame(results)
        df_results.to_csv("./data/optimization_results/all_experiments.csv", 
                         index=False, encoding='utf-8')
        
        # Сохраняем в JSON для детального анализа
        with open("./data/optimization_results/detailed_results.json", "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Результаты сохранены в ./data/optimization_results/")
    
    def analyze_results(self, results: List[Dict]):
        """Анализ и визуализация результатов"""
        if not results:
            print("❌ Нет результатов для анализа")
            return
        
        df = pd.DataFrame(results)
        
        # Находим абсолютно лучшую комбинацию
        best_overall = df.loc[df['MRR'].idxmax()]
        
        print(f"\n🎯 АБСОЛЮТНО ЛУЧШАЯ КОМБИНАЦИЯ:")
        print(f"Стратегия: {best_overall['strategy']}")
        print(f"Параметры: {best_overall['params']}")
        print(f"MRR: {best_overall['MRR']:.3f}")
        print(f"Hits@1: {best_overall['Hits@1']:.3f}")
        print(f"Chunks: {best_overall['total_chunks']}")
        print(f"Средняя длина: {best_overall['avg_chunk_words']:.1f} слов")
        
        # Анализ по стратегиям
        print(f"\n📊 СРАВНЕНИЕ СТРАТЕГИЙ (лучшие результаты):")
        best_by_strategy = df.loc[df.groupby('strategy')['MRR'].idxmax()]
        
        for _, row in best_by_strategy.iterrows():
            print(f"{row['strategy']:12} | MRR: {row['MRR']:.3f} | "
                  f"Hits@1: {row['Hits@1']:.3f} | Chunks: {row['total_chunks']:4} | "
                  f"Words: {row['avg_chunk_words']:5.1f}")
        
        # Анализ зависимости качества от размера chunks
        print(f"\n📈 ЗАВИСИМОСТЬ КАЧЕСТВА ОТ РАЗМЕРА CHUNKS:")
        df_sorted = df.sort_values('avg_chunk_words')
        for _, row in df_sorted.iterrows():
            if row['avg_chunk_words'] > 0:
                print(f"{row['avg_chunk_words']:5.1f} слов | MRR: {row['MRR']:.3f} | "
                      f"{row['strategy']} {row['params']}")
        
        # Рекомендации
        self.generate_recommendations(df)
    
    def generate_recommendations(self, df: pd.DataFrame):
        """Генерация рекомендаций на основе результатов"""
        print(f"\n💡 РЕКОМЕНДАЦИИ:")
        
        # Лучшая стратегия
        best = df.loc[df['MRR'].idxmax()]
        print(f"1. Используйте стратегию: {best['strategy']}")
        print(f"2. С параметрами: {best['params']}")
        
        # Анализ эффективности
        efficient = df[df['total_chunks'] < 500].nlargest(3, 'MRR')
        if len(efficient) > 0:
            print(f"3. Эффективные альтернативы (мало chunks, высокий MRR):")
            for _, row in efficient.iterrows():
                print(f"   - {row['strategy']}: MRR {row['MRR']:.3f}, {row['total_chunks']} chunks")
        
        # Общие закономерности
        avg_words_best = df.nlargest(5, 'MRR')['avg_chunk_words'].mean()
        print(f"4. Оптимальный размер chunks: {avg_words_best:.1f} слов в среднем")

def main():
    """Основная функция"""
    print("🚀 AUTOMATIC CHUNK OPTIMIZATION SYSTEM")
    print("=" * 60)
    
    # Загрузка данных
    print("Загрузка данных...")
    docs = pd.read_csv("./data/dataset/texts.csv")
    questions = pd.read_csv("./data/dataset/questions.csv")
    
    # Загрузка текстов документов
    print("Загрузка текстов документов...")
    raw_texts = []
    for index, row in docs.iterrows():
        try:
            file_path = f"./data/dataset/texts/{row['page_id']}.txt"
            with open(file_path, "r", encoding='utf-8') as f:
                raw_texts.append(f.read())
        except FileNotFoundError:
            raw_texts.append("")
    
    docs["text"] = raw_texts
    
    print(f"Загружено: {len(docs)} документов, {len(questions)} вопросов")
    
    # Создаем оптимизатор
    optimizer = ChunkOptimizer(docs, questions)
    
    # Запускаем оптимизацию
    start_time = time.time()
    all_results = optimizer.run_comprehensive_optimization()
    end_time = time.time()
    
    print(f"\n⏱️  Общее время выполнения: {(end_time - start_time)/60:.1f} минут")
    print(f"📊 Проведено экспериментов: {len(all_results)}")
    
    # Сохраняем финальный отчет
    if all_results:
        best_result = max(all_results, key=lambda x: x['MRR'])
        report = {
            "best_strategy": best_result['strategy'],
            "best_params": best_result['params'],
            "best_mrr": best_result['MRR'],
            "best_hits1": best_result['Hits@1'],
            "total_experiments": len(all_results),
            "optimization_time_minutes": (end_time - start_time)/60
        }
        
        with open("./data/optimization_results/final_report.json", "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ ОПТИМИЗАЦИЯ ЗАВЕРШЕНА!")
        print(f"🎯 РЕКОМЕНДУЕМАЯ НАСТРОЙКА:")
        print(f"   Стратегия: {best_result['strategy']}")
        print(f"   Параметры: {best_result['params']}")
        print(f"   Ожидаемый MRR: {best_result['MRR']:.3f}")

if __name__ == "__main__":
    main()