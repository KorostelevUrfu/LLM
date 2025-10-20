import pandas as pd
import re
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer
import os
import shutil

def setup_environment():
    """Подготовка окружения и загрузка данных"""
    print("Настройка окружения...")
    
    # Создаем директории
    os.makedirs("./data/dataset/texts", exist_ok=True)
    os.makedirs("./data/results", exist_ok=True)
    
    # Проверяем существование файлов
    required_files = [
        "./data/dataset/questions.csv",
        "./data/dataset/texts.csv",
        "./data/dataset/texts/1.txt",
        "./data/dataset/texts/2.txt",
        "./data/dataset/texts/3.txt",
        "./data/dataset/texts/5.txt"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Ошибка: Отсутствуют файлы: {missing_files}")
        print("Убедитесь, что вы загрузили все файлы датасета")
        return False
    
    return True

def load_data():
    """Загрузка данных"""
    print("Загрузка данных...")
    
    try:
        # Загрузка вопросов
        q = pd.read_csv("./data/dataset/questions.csv")
        print(f"Загружено вопросов: {len(q)}")
        
        # Загрузка метаданных документов
        docs = pd.read_csv("./data/dataset/texts.csv")
        print(f"Загружено документов: {len(docs)}")
        
        # Загрузка текстов документов
        raw_texts = []
        successful_loads = 0
        
        for index, row in docs.iterrows():
            try:
                file_path = f"./data/dataset/texts/{row['page_id']}.txt"
                with open(file_path, "r", encoding='utf-8') as f:
                    text = f.read()
                    raw_texts.append(text)
                    successful_loads += 1
            except FileNotFoundError:
                print(f"Предупреждение: Файл {file_path} не найден")
                raw_texts.append("")  # Добавляем пустой текст
        
        docs["text"] = raw_texts
        print(f"Успешно загружено текстов: {successful_loads}/{len(docs)}")
        
        return q, docs
        
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None, None

# Функции для разбиения документов (остаются без изменений)
def split_by_sentences(text: str) -> List[str]:
    """Разбиение по предложениям"""
    if not text or len(text.strip()) == 0:
        return []
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
    return sentences

def split_by_paragraphs(text: str) -> List[str]:
    """Разбиение по абзацам"""
    if not text or len(text.strip()) == 0:
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]
    return paragraphs

def split_by_fixed_length(text: str, max_words: int = 100) -> List[str]:
    """Разбиение на куски фиксированной длины"""
    if not text or len(text.strip()) == 0:
        return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def split_with_overlap(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    """Разбиение с перекрытием"""
    if not text or len(text.strip()) == 0:
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def split_by_headings(text: str) -> List[str]:
    """Разбиение по заголовкам (h1, h2, h3)"""
    if not text or len(text.strip()) == 0:
        return []
    
    # Регулярное выражение для поиска заголовков
    heading_pattern = r'(^h\d\.\s+.*?)(?=\n\n|$|\\nh\d\.\s+)'
    chunks = []
    
    # Находим все заголовки
    matches = list(re.finditer(heading_pattern, text, re.DOTALL | re.MULTILINE))
    
    if not matches:
        # Если заголовков нет, разбиваем по абзацам
        return split_by_paragraphs(text)
    
    for i, match in enumerate(matches):
        start_pos = match.start()
        
        if i < len(matches) - 1:
            # Текст от текущего заголовка до следующего
            end_pos = matches[i + 1].start()
            chunk_content = text[start_pos:end_pos].strip()
        else:
            # Последний chunk - до конца текста
            chunk_content = text[start_pos:].strip()
        
        if len(chunk_content) > 10:
            chunks.append(chunk_content)
    
    return chunks

def create_chunks_dataset(docs_df, mode="paragraphs"):
    """Создает датасет из разбитых документов"""
    chunks_data = []
    
    for index, row in docs_df.iterrows():
        text = row["text"]
        page_id = row["page_id"]
        title = row["title"]
        
        # Пропускаем пустые тексты
        if not text or len(text.strip()) == 0:
            continue
            
        if mode == "sentences":
            chunks = split_by_sentences(text)
        elif mode == "paragraphs":
            chunks = split_by_paragraphs(text)
        elif mode == "fixed":
            chunks = split_by_fixed_length(text, max_words=150)
        elif mode == "overlap":
            chunks = split_with_overlap(text, chunk_size=150, overlap=30)
        elif mode == "headings":
            chunks = split_by_headings(text)
        else:
            chunks = split_by_paragraphs(text)  # по умолчанию
        
        for chunk_idx, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) > 10:  # Игнорируем слишком короткие chunks
                chunks_data.append({
                    "text": chunk_text,
                    "page_id": page_id,
                    "title": title,
                    "chunk_id": f"{page_id}_{chunk_idx}",
                    "chunk_index": chunk_idx
                })
    
    return pd.DataFrame(chunks_data)

def evaluate_strategy(chunks_df, strategy_name, q, top_k=5):
    """Оценка качества поиска для конкретной стратегии"""
    
    # Создаем новую коллекцию в ChromaDB
    client = chromadb.PersistentClient(f"./data/chroma_{strategy_name}")
    
    # Очищаем старую коллекцию если существует
    try:
        client.delete_collection(strategy_name)
    except:
        pass
    
    collection = client.get_or_create_collection(
        name=strategy_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Инициализация модели эмбеддингов
    print("  Загрузка модели эмбеддингов...")
    model = SentenceTransformer("ai-forever/ru-en-RoSBERTa")
    
    # Добавляем chunks в векторную базу
    print(f"  Добавление {len(chunks_df)} chunks в векторную базу...")
    
    # Добавляем пачками для эффективности
    batch_size = 50
    for i in range(0, len(chunks_df), batch_size):
        batch = chunks_df.iloc[i:i+batch_size]
        
        texts = batch["text"].tolist()
        embeddings = model.encode(texts)
        metadatas = [{"page_id": row["page_id"], "chunk_id": row["chunk_id"]} 
                    for _, row in batch.iterrows()]
        ids = [f"id_{i+j}" for j in range(len(batch))]
        
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
    
    # Оценка MRR
    print("  Оценка MRR...")
    df_data = []
    
    for index, row in q.iterrows():
        question = row["question"]
        target_page_id = row["page_id"]
        query = model.encode(question)
        
        results = collection.query(query_embeddings=[query.tolist()], n_results=top_k)
        
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
    
    # Расчет MRR
    df = pd.DataFrame(df_data, columns=["question", "position", "score"])
    df["rank"] = df["position"].apply(lambda x: 1 / x if x != 0 else 0)
    MRR = df["rank"].sum() / len(q)
    
    # Дополнительные метрики
    hits_at_1 = len(df[df["position"] == 1])
    hits_at_3 = len(df[df["position"] <= 3])
    hits_at_5 = len(df[df["position"] <= 5])
    
    return {
        "MRR": MRR,
        "Hits@1": hits_at_1 / len(q),
        "Hits@3": hits_at_3 / len(q), 
        "Hits@5": hits_at_5 / len(q),
        "details": df
    }

def main():
    """Основная функция"""
    print("=" * 60)
    print("RAG SYSTEM - ОПТИМИЗАЦИЯ РАЗБИЕНИЯ ДОКУМЕНТОВ")
    print("=" * 60)
    
    # Подготовка окружения
    if not setup_environment():
        return
    
    # Загрузка данных
    q, docs = load_data()
    if q is None or docs is None:
        print("Не удалось загрузить данные. Завершение работы.")
        return
    
    print(f"\nДанные успешно загружены:")
    print(f"  - Вопросов: {len(q)}")
    print(f"  - Документов: {len(docs)}")
    
    # Создаем chunks для всех стратегий
    strategies = ["paragraphs", "sentences", "fixed", "overlap", "headings"]
    chunks_datasets = {}
    
    print(f"\nСоздание chunks для разных стратегий...")
    for strategy in strategies:
        print(f"Обработка стратегии: {strategy}")
        chunks_df = create_chunks_dataset(docs, mode=strategy)
        chunks_datasets[strategy] = chunks_df
        print(f"  - Создано {len(chunks_df)} chunks")
    
    # Оценка всех стратегий
    print(f"\nОценка всех стратегий разбиения...")
    results = {}
    
    for strategy in strategies:
        print(f"\nОценка стратегии: {strategy}")
        chunks_df = chunks_datasets[strategy]
        result = evaluate_strategy(chunks_df, strategy, q)
        results[strategy] = result
        print(f"  MRR: {result['MRR']:.3f}")
        print(f"  Hits@1: {result['Hits@1']:.3f}")
        print(f"  Hits@3: {result['Hits@3']:.3f}")
        print(f"  Hits@5: {result['Hits@5']:.3f}")
    
    # Сравнительная таблица результатов
    print("\n" + "=" * 60)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    comparison_data = []
    for strategy in strategies:
        result = results[strategy]
        comparison_data.append({
            "Strategy": strategy,
            "MRR": f"{result['MRR']:.3f}",
            "Hits@1": f"{result['Hits@1']:.3f}",
            "Hits@3": f"{result['Hits@3']:.3f}",
            "Hits@5": f"{result['Hits@5']:.3f}",
            "Chunks Count": len(chunks_datasets[strategy])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Находим лучшую стратегию
    best_strategy = max(results.keys(), key=lambda x: results[x]['MRR'])
    best_result = results[best_strategy]
    
    print(f"\n🎉 ЛУЧШАЯ СТРАТЕГИЯ: {best_strategy}")
    print(f"📊 MRR: {best_result['MRR']:.3f}")
    print(f"🎯 Hits@1: {best_result['Hits@1']:.3f}")
    
    # Сохраняем результаты
    output_dir = "./data/results"
    
    # Сохраняем сравнение
    comparison_df.to_csv(f"{output_dir}/strategies_comparison.csv", index=False, encoding='utf-8')
    
    # Сохраняем детали лучшей стратегии
    best_details = results[best_strategy]['details']
    best_details.to_csv(f"{output_dir}/best_strategy_{best_strategy}_details.csv", index=False, encoding='utf-8')
    
    print(f"\n💾 Результаты сохранены в директорию: {output_dir}")
    
    # Дополнительная информация о chunks
    print(f"\n📈 СТАТИСТИКА ПО CHUNKS:")
    for strategy in strategies:
        chunks_df = chunks_datasets[strategy]
        if len(chunks_df) > 0:
            avg_length = chunks_df['text'].str.split().str.len().mean()
        else:
            avg_length = 0
        print(f"{strategy:12} | {len(chunks_df):4} chunks | Средняя длина: {avg_length:.1f} слов")
    
    print(f"\n✅ Завершено успешно!")

if __name__ == "__main__":
    main()