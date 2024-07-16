import streamlit as st
import os
from litellm import completion
import PyPDF2
from docx import Document
import tiktoken
import nltk
import ssl
import docx2txt

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

st.sidebar.markdown("""
**Coded by:** Tseng Yao Hsien 
**Contact:** zinojeng@gmail.com  
**Reference:** Andrew Ng's AI Agent System for Language Translation [https://github.com/andrewyng/translation-agent](https://github.com/andrewyng/translation-agent)

<small>Translation Agent 是一個基於 Python 的開源專案，它採用了一種新穎的“省思式工作流程"，旨在解決傳統機器翻譯系統在定制化和細粒度控制方面的不足。</small>
<small>Translation Agent 的工作原理：省思式工作流程 Translation Agent 的核心在於其獨特的“省思式工作流程”，該工作流程模擬了人類翻譯專家的思考過程，將翻譯任務分解為三個主要步驟：</small>
<small>1. 初始翻譯：Translation Agent 首先利用 LLM 對輸入文本進行初步翻譯，產出一個初譯的內容。</small>
<small>2. 反思與改進：與傳統機器翻譯系統直接輸出譯文不同，Translation Agent 會引導 LLM 對自身的翻譯結果進行反思，並提出改進建議。例如，LLM 可能會指出譯文中存在的不準確、不流暢或不符合目標語言的慣用表達的地方，就像一個語言審核者一樣，幫助 LLM 找出翻譯中的不足之處。</small>
<small>3. 優化輸出：最後，根據 LLM 的建議，Translation Agent 對初譯的內容進行優化處理，產生更符合目標語言慣用表達、更精確且流暢的譯文。</small>
""")
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
st.empty()  # 添加这行来清除可能的缓存

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
            text = docx2txt.process(file)
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
    uploaded_file = st.file_uploader("Choose a Word Document", type=["docx"])
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
    encoding = tiktoken.encoding_for_model("gpt-4")
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
    system_message = f"You are an expert medical translator, specializing in translating medical instructions and educational materials from {source_lang} to {target_lang}."
    translation_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""
    return get_completion(translation_prompt, system_message=system_message, model=model)

def one_chunk_reflect_on_translation(model, source_text, translation_1):
    system_message = f"You are an expert medical translator specializing in translation from {source_lang} to {target_lang}. \
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
        result = one_chunk_translate_text("gpt-4", source_text)
    
    if result is None:
        st.error("Translation failed. Please try again.")
        return
    
    st.success("Translation completed!")
    
    # Prepare download button
    result_text = f"""Source Text:
{source_text}

Improved Translation:
{result['improved_translation']}

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