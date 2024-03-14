import argparse
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import SentenceTransformer
import faiss
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers.utils import logging


LANGCHAIN_CHUNK_SIZE = 1000
LANGCHAIN_K_VALUE = 3
LANGCHAIN_CHUNK_OVERLAP = 100

CUSTOM_CHUNK_SIZE = 768
CUSTOM_K_VALUE = 3
CUSTOM_CHUNK_OVERLAP = 100

def cosine_similarity_search(query_embedding, passage_embeddings, k):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), passage_embeddings)[0]
    indices = similarities.argsort()[::-1][:k]
    return indices

def dot_product_search(query_embedding, passage_embeddings, k=1):
    similarities = np.dot(query_embedding, passage_embeddings.T)
    indices = similarities.argsort()[::-1][:k]
    return indices

def euclidean_distance_search(query_embedding, passage_embeddings, k=1):
    distances = np.linalg.norm(passage_embeddings - query_embedding, axis=1)
    indices = distances.argsort()[:k]
    return indices

def qa_no_rag(questions, args):
    '''
    Basic QA system without RAG
    
    questions: pandas dataframe with 'question' column
    args: argparse object with 'output' attribute
    
    '''
    
    print("Running QA without RAG..")

    # Load the model
    model = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 100},
        device=0
    )
    
    # Iterate through each question and get the answer
    answers = []
    for row in questions.iterrows():
        outputs = model.invoke(f"Question: {row[1]['question']} \nAnswer: ")
        answers.append(outputs)
        
    # Save the questions, answers to a CSV file
    df = pd.DataFrame({'question': questions['question'].tolist(), 'answer': answers})
    df.to_csv(args.output, index=False)
    
    print('Done writing to file: ', args.output)
    

def qa_rag_with_langchain(questions, args):
    '''
    QA system with RAG and LangChain embeddings
    
    questions: pandas dataframe with 'question' column
    args: argparse object with 'output' attribute
    
    '''
    
    print("Running QA with RAG and LangChain embeddings...")

    model = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
        pipeline_kwargs={"max_new_tokens": 256},
        device=0
    )

    csvloader = CSVLoader(file_path=args.passages, source_column="context")
    passages_csv_data = csvloader.load()
    embeddings = HuggingFaceEmbeddings()

    text_splitter = CharacterTextSplitter(chunk_size=LANGCHAIN_CHUNK_SIZE, chunk_overlap=LANGCHAIN_CHUNK_OVERLAP)
    docs = text_splitter.split_documents(passages_csv_data)
    db = FAISS.from_documents(docs, embeddings)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    answers = []
    top_k_docs = []
    for idx, row in questions.iterrows():
        template = "Answer the following question given the context \n question: {question} \n context: {context}"
        
        prompt = PromptTemplate.from_template(template)


        retriever=db.as_retriever(search_kwargs={"k": LANGCHAIN_K_VALUE})
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        answers.append(rag_chain.invoke(row['question']))
        top_k_docs.append([i.metadata['source'] for i in retriever.get_relevant_documents(row['question'])])

    df = pd.DataFrame({'question': questions['question'].tolist(), 'answer': answers, 'sources': top_k_docs})
    df.to_csv(args.output, index=False)
    
    print('Done writing to file: ', args.output)
        
        
def qa_rag_with_custom_embeddings(questions, args):
    '''
    QA system with RAG and custom embeddings
    
    questions: pandas dataframe with 'question' column
    args: argparse object with 'output' attribute
    
    '''
    
    print("Running QA with RAG and custom embeddings...")
    
    df_passage = pd.read_csv(args.passages)

    tokenize_model = SentenceTransformer('sentence-transformers/roberta-base-nli-stsb-mean-tokens')

    embeddings = []
    for idx, row in df_passage.iterrows():
        passage_embedding = tokenize_model.encode(row['context'])
        embeddings.append(passage_embedding)
    embeddings_np = np.array(embeddings) / np.linalg.norm(np.array(embeddings))
    
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)

    def retrieve_documents(query, k=CUSTOM_K_VALUE, similarity_measure=None):
        query_embedding = tokenize_model.encode(query)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize query embedding

        if similarity_measure == 'cosine_similarity':
            indices = cosine_similarity_search(query_embedding, embeddings_np, k)
            retrieved_docs = df_passage.iloc[indices]['context'].tolist()
            return retrieved_docs
        elif similarity_measure == 'dot_product':
            indices = dot_product_search(query_embedding, embeddings_np, k)
            retrieved_docs = df_passage.iloc[indices]['context'].tolist()
            return retrieved_docs
        elif similarity_measure == 'euclidean_distance':
            indices = euclidean_distance_search(query_embedding, embeddings_np, k)
            retrieved_docs = df_passage.iloc[indices]['context'].tolist()
            return retrieved_docs
        else:
            query_embedding = tokenize_model.encode(query)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = np.expand_dims(query_embedding, axis=0)  # Add batch dimension

            _, indices = index.search(query_embedding, k)
            retrieved_docs = df_passage.iloc[indices[0]]['context'].tolist()
            
            return retrieved_docs

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map="auto")

    def generate_answer(question, retrieved_docs, max_length=500):
        # concatenate retrieved documents for context
        context = "\n".join(retrieved_docs)
        # Generate answer 
        input_text = f"Question: {question} \n context: {context}"

        input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids.to(model.device)
        outputs = model.generate(input_ids, max_length=max_length, temperature=0.8, do_sample=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    

    answers = []
    top_k_docs = []
    for idx, row in questions.iterrows():
        retrieved_docs = retrieve_documents(row['question'], CUSTOM_K_VALUE, similarity_measure=None)
        answer = generate_answer(row['question'], retrieved_docs)
        answers.append(answer)
        top_k_docs.append(retrieved_docs)

    df = pd.DataFrame({'question': questions['question'].tolist(), 'answer': answers, 'sources': top_k_docs})
    df.to_csv(args.output, index=False)
    
    print('Done writing to file: ', args.output)
    

def main():
    
    parser = argparse.ArgumentParser(description='Question answering using RAG and LangChain/custom embeddings.')

    # Required arguments
    parser.add_argument('--questions', type=str, required=True, help='Path to questions CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path to output CSV file')

    # Optional flags
    parser.add_argument('--rag', action='store_true', help='Enable RAG')
    parser.add_argument('--langchain', action='store_true', help='Use langchain embeddings')
    parser.add_argument('--passages', type=str, help='Path to passages CSV file')

    args = parser.parse_args()
    
    df_question = pd.read_csv(args.questions)

    if args.rag and args.langchain:
        qa_rag_with_langchain(df_question, args)

    elif args.rag:
        qa_rag_with_custom_embeddings(df_question, args)
        
    else:
        qa_no_rag(df_question, args)
        

if __name__ == "__main__":
    logging.set_verbosity(40)
    main()