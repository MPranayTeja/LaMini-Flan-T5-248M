import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

#model and tokenizer loading
# checkpoint = "LaMini-Flan-T5-248M"
# tokenizer = T5Tokenizer.from_pretrained(checkpoint)
# base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

model_name = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

#file loader and preprocessing
def file_preprocessing(file):
    loader =  PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        print(text)
        final_texts = final_texts + text.page_content
    return final_texts

#LLM pipeline
# def llm_pipeline(filepath):
#     pipe_sum = pipeline(
#         'summarization',
#         model = base_model,
#         tokenizer = tokenizer,
#         max_length = 883, 
#         min_length = 50)
#     input_text = file_preprocessing(filepath)
#     result = pipe_sum(input_text)
#     result = result[0]['summary_text']
#     return result

# @st.cache_data
#function to display the PDF of a given file 
# def displayPDF(file):
#     with open(file, "rb") as f:
#         base64_pdf = base64.b64encode(f.read()).decode('utf-8')

#     pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

#     st.markdown(pdf_display, unsafe_allow_html=True)
def summarization(text_to_summarize):
    inputs = tokenizer.encode(text_to_summarize, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs, max_length=500, min_length=200, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
#streamlit code 
st.set_page_config(layout="wide")
def main():
    st.title("Text Summarization App using Language Model")

    # Text input field instead of file uploader
    text_input = st.text_area("Enter the text to summarize")

    if text_input and st.button("Summarize"):
        col1, col2 = st.columns(2)
        with col1:
            st.info("Input Text")
            st.write(text_input)  # Display the input text
        with col2:
            # Perform summarization on the input text
            summary=summarization(text_input)
            st.info("Summarization Complete")
            st.success(summary)        


if _name_ == "_main_":
    main()