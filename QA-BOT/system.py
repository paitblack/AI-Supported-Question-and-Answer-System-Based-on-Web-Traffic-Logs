from log_vectorizer import LogVectorizer
from response_generator import ResponseGenerator
import numpy as np
from langchain.embeddings import HuggingFaceBgeEmbeddings

class System():
    def __init__(self, text_file="data.txt", index_file="faiss_index", model_name="t5-base"):
        self.vectorizer = LogVectorizer(text_file=text_file, index_file=index_file)  #to handling log data and FAISS indexing.
        self.gen_response = ResponseGenerator(model_name=model_name)  #to be able to answer.
    
    def process_and_index_data(self, df):
        self.vectorizer.convert_it_to_text(df)  #convert dataframe's text column to a text file.
        documents = self.vectorizer.split_it()  #split them into smaller parts.
        
        self.vectorizer.create_and_save_faiss_index(documents) #create the FAISS and save it.

    def give_us_answer(self, question, top_k=1):
        vectorDB = self.vectorizer.load_faiss_index()  #load the FAISS
        embedding_model = HuggingFaceBgeEmbeddings()  #initialize the embedding model.
        
        question_embedding = embedding_model.embed_documents([question])[0]
        question_embedding = np.array(question_embedding).astype(np.float32).reshape(1, -1)  #reshaped for the correct size
        
        distances, indices = vectorDB.search(question_embedding, top_k)  # search FAISS to get closest matches
        
        documents = self.vectorizer.split_it()   
        if indices[0][0] >= len(documents):   #check if the index is valid.
            context = "No relevant context found."
        else:
            context = documents[indices[0][0]].page_content #retrive the closest matched.
        
        response = self.gen_response.generate_response(question=question, context=context)  #generated response.
        return response

