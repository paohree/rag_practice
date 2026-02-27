import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

PERSIST_DIR = "./chroma_store"

def build_or_load_vectordb():
    embeddings = OpenAIEmbeddings()

    # 이미 만들어둔 벡터DB가 있으면 로드
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    # 없으면 docs 폴더의 txt 문서를 읽어서 새로 생성
    loader = DirectoryLoader(
        "docs",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    return vectordb

def main():
    vectordb = build_or_load_vectordb()
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    print("RAG ready. docs/에 txt 넣고 질문하세요. 종료: q")

    while True:
        q = input("\n질문: ").strip()
        if q.lower() == "q":
            break

        sources = retriever.invoke(q)
        context = "\n\n".join(
            [f"[{i+1}] ({d.metadata.get('source','unknown')})\n{d.page_content[:500]}"
             for i, d in enumerate(sources)]
        )

        prompt = f"""너는 문서 기반 분석가다.
아래 CONTEXT에 있는 내용만 근거로 답해라.
모르면 모른다고 말해라.
답변 마지막에 근거 번호([1],[2]...)를 꼭 붙여라.

QUESTION:
{q}

CONTEXT:
{context}
"""

        ans = llm.invoke(prompt).content
        print("\n[답변]\n", ans)

        print("\n[근거 조각]")
        for i, d in enumerate(sources, 1):
            print(f"\n--- [{i}] {d.metadata.get('source','unknown')} ---")
            print(d.page_content[:500])

if __name__ == "__main__":
    main()