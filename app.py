import streamlit as st
import os
from litellm import completion
import PyPDF2
import io
from docx import Document
import re
import tiktoken
import nltk
import ssl
import textract

# SSL and NLTK setup
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)


# Set page config
st.set_page_config(page_title="Translation Agent", layout="wide")

# Sidebar for API key input
st.sidebar.title("Configuration")
openai_api_key = st.sidebar.text_input(
    label="Enter your OpenAI API Key:",
    type='password',
    placeholder="Ex: sk-2twmA88un4...",
    help="You can get your API key from https://platform.openai.com/account/api-keys/"
)

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Language selection
st.header("Select Languages")
col1, col2, col3 = st.columns(3)

with col1:
    source_lang = st.selectbox(
        "Source Language",
        ["English", "Chinese", "Spanish", "French", "German", "Italian", "Japanese", "Korean", "Vietnamese", "Indonesian", "Thai"]
    )

with col2:
    target_lang = st.selectbox(
        "Target Language",
        ["Traditional Chinese", "Simplified Chinese", "English", "Spanish", "French", "German", "Italian", "Japanese", "Korean", "Vietnamese", "Indonesian", "Thai"]
    )

with col3:
    country_options = {
        "Traditional Chinese": ["Taiwan", "Hong Kong"],
        "Simplified Chinese": ["China", "Singapore"],
        "English": ["USA", "UK", "Australia", "Canada"],
        "Spanish": ["Spain", "Mexico", "Argentina"],
        "French": ["France", "Canada", "Belgium"],
        "German": ["Germany", "Austria", "Switzerland"],
        "Italian": ["Italy", "Switzerland"],
        "Japanese": ["Japan"],
        "Korean": ["South Korea"],
        "Vietnamese": ["Vietnam"],
        "Indonesian": ["Indonesia"],
        "Thai": ["Thailand"]
    }
    country = st.selectbox("Country/Region", country_options.get(target_lang, []))

# Input method selection
input_method = st.radio("Choose input method:", ("Enter Text", "Upload PDF", "Upload TXT", "Upload Word Document"))

# Function to read PDF
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to read TXT
def read_txt(file):
    return file.getvalue().decode("utf-8")

# Function to read Word Document
def read_doc_or_docx(file):
    file_extension = file.name.split('.')[-1].lower()
    try:
        if file_extension == 'docx':
            doc = Document(file)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return '\n'.join(full_text)
        elif file_extension == 'doc':
            # 将文件内容保存到临时的字节流中
            bytes_io = io.BytesIO(file.getvalue())
            # 使用 textract 读取 .doc 文件
            text = textract.process(bytes_io, extension='doc').decode('utf-8')
            return text
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return ""

# Input text based on selected method
if input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        source_text = read_pdf(uploaded_file)
        st.text_area("Extracted text from PDF:", value=source_text, height=200)
    else:
        source_text = ""
elif input_method == "Upload TXT":
    uploaded_file = st.file_uploader("Choose a TXT file", type="txt")
    if uploaded_file is not None:
        source_text = read_txt(uploaded_file)
        st.text_area("Extracted text from TXT:", value=source_text, height=200)
    else:
        source_text = ""
elif input_method == "Upload Word Document":
    uploaded_file = st.file_uploader("Choose a Word Document", type=["doc", "docx"])
    if uploaded_file is not None:
        source_text = read_doc_or_docx(uploaded_file)
        if source_text:
            st.text_area("Extracted text from Word Document:", value=source_text, height=200)
        else:
            st.error("Failed to extract text from the document.")
    else:
        source_text = ""
else:  # Enter Text
    source_text = st.text_area("Enter the text to translate:", height=200)

def estimate_token_count(text):
    encoding = tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(text))

def estimate_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1_000_000) * 5.00
    output_cost = (output_tokens / 1_000_000) * 15.00
    total_cost_usd = input_cost + output_cost
    return total_cost_usd * 30  # Assuming 1 USD = 30 NTD


# Translation functions
def get_completion(user_prompt, system_message="You are a helpful assistant.", model="gpt-4o", temperature=0.3):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=model, messages=messages)
    return response["choices"][0]["message"]["content"]

def one_chunk_initial_translation(model, source_text):
    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."
    translation_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""
    return get_completion(translation_prompt, system_message=system_message, model=model)

def one_chunk_reflect_on_translation(model, source_text, translation_1):
    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."
    prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and takes into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""
    return get_completion(prompt, system_message=system_message, model=model)

def one_chunk_improve_translation(model, source_text, translation_1, reflection):
    system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."
    prompt = f"""Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Provide your improved translation as a continuous text, without any additional formatting or labels."""

    return get_completion(prompt, system_message, model=model)

def create_sentence_pairs(source_text, translated_text):
    source_sentences = nltk.sent_tokenize(source_text)
    translated_sentences = nltk.sent_tokenize(translated_text)

    sentence_pairs = []
    for i in range(min(len(source_sentences), len(translated_sentences))):
        source_sentence = source_sentences[i].strip()
        translated_sentence = translated_sentences[i].strip()
        if source_sentence and translated_sentence:
            sentence_pairs.append({
                "Original": source_sentence,
                "Translation": translated_sentence
            })
    return sentence_pairs

def one_chunk_translate_text(model, source_text):
    try:
        st.subheader("Initial Translation")
        translation_1 = one_chunk_initial_translation(model, source_text)
        st.write(translation_1)

        st.subheader("Translation Reflection")
        reflection = one_chunk_reflect_on_translation(model, source_text, translation_1)
        st.write(reflection)

        st.subheader("Improved Translation")
        improved_translation = one_chunk_improve_translation(model, source_text, translation_1, reflection)
        st.write(improved_translation)

        st.subheader("Sentence-by-Sentence Comparison")
        sentence_pairs = create_sentence_pairs(source_text, improved_translation)

        if sentence_pairs:
            for pair in sentence_pairs:
                st.write(f"Original: {pair['Original']}")
                st.write(f"Translation: {pair['Translation']}\n")  # 顯示每一句的翻譯結果
        else:
            st.write("No sentence pairs could be generated.")
        
        input_tokens = estimate_token_count(source_text)
        output_tokens = estimate_token_count(translation_1) + estimate_token_count(reflection) + estimate_token_count(improved_translation)
        total_tokens = input_tokens + output_tokens
        estimated_cost = estimate_cost(input_tokens, output_tokens)

        st.subheader("Token Usage and Cost Estimation")
        st.write(f"Total tokens used: {total_tokens}")
        st.write(f"Input tokens: {input_tokens}")
        st.write(f"Output tokens: {output_tokens}")
        st.write(f"Estimated cost: NTD {estimated_cost:.2f}")

        return {
            "initial_translation": translation_1,
            "reflection": reflection,
            "improved_translation": improved_translation,
            "sentence_pairs": sentence_pairs,
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": estimated_cost
        }
    except Exception as e:
        st.error(f"An error occurred during translation processing: {str(e)}")
        return None

# Translate button
def perform_translation():
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        return
    
    if not source_text:
        st.error("Please provide some text to translate.")
        return
    
    with st.spinner("Translating... This may take a moment."):
        result = one_chunk_translate_text("gpt-4o", source_text)
    
    if result is None:
        st.error("Translation failed. Please try again.")
        return
    
    st.success("Translation completed!")
    
    # Prepare download button
    result_text = f"""Source Text:
{source_text}

Initial Translation:
{result['initial_translation']}

Translation Reflection:
{result['reflection']}

Improved Translation:
{result['improved_translation']}

Sentence-by-Sentence Comparison:
"""
    for pair in result['sentence_pairs']:
        result_text += f"Original: {pair['Original']}\nTranslation: {pair['Translation']}\n\n"

    result_text += f"""Token Usage:
Total tokens: {result['total_tokens']}
Input tokens: {result['input_tokens']}
Output tokens: {result['output_tokens']}

Estimated Cost: NTD {result['estimated_cost']:.2f}
"""

    st.download_button(
        label="Download Translation Results",
        data=result_text,
        file_name="translation_results.txt",
        mime="text/plain"
    )

if st.button("Translate"):
    perform_translation()
    st.info("Execution finished")