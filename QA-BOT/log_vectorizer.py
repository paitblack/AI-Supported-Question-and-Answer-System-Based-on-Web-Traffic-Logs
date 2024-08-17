import os
import pickle
import faiss
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings

class LogVectorizer():
    def __init__(self, text_file="data.txt", index_file="faiss_index"):
        self.text_file = text_file   #name of the file where text will be saved.
        self.index_file = index_file   # FAISS will be saved.

    def convert_it_to_text(self, df):   #to write the texts in df into a new text file
        if 'text' in df.columns:
            with open(self.text_file, "w") as file:
                for text in df["text"]:
                    file.write(str(text) + "\n")
        else:
            print("The 'text' column does not exist in the DataFrame.")  #to fit all kind of df

    def split_it(self, path=None):
        if path is None:
            path = self.text_file
            
        loader = TextLoader(path)        #get text with its path and load
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,   #chunk size declared as 250, tried from 150 to 300 and decided 250 fitted best.
            chunk_overlap=0,
            length_function=len
        )

        documents = text_splitter.split_documents(data)  
        return documents   #It returns the splitted document.

    def create_and_save_faiss_index(self, documents, embedding_model=HuggingFaceBgeEmbeddings):
        embeddings = embedding_model()
        faiss_index = FAISS.from_documents(documents, embeddings)
        
        faiss.write_index(faiss_index.index, self.index_file)  #get a new file to save the FAISS index, so dont need to process the data again.

    def load_faiss_index(self):
        index = faiss.read_index(self.index_file)
        return index   # loadd the saved FAISS.

