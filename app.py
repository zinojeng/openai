import streamlit as st
import os
from litellm import completion
import PyPDF2
import io
from docx import Document
import re

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
input_method = st.radio("Choose input method:", ("Upload PDF", "Upload TXT", "Upload Word Document", "Enter Text"))

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
def read_docx(file):
    doc = Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

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
        source_text = read_docx(uploaded_file)
        st.text_area("Extracted text from Word Document:", value=source_text, height=200)
    else:
        source_text = ""
else:  # Enter Text
    source_text = st.text_area("Enter the text to translate:", height=200)

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

Provide your improved translation in the following format:
[SOURCE] Original sentence 1
[TARGET] Improved translation of sentence 1

[SOURCE] Original sentence 2
[TARGET] Improved translation of sentence 2

... and so on for each sentence or logical unit of the text.

After providing the sentence-by-sentence translation, please also provide a full, continuous improved translation of the entire text."""

    return get_completion(prompt, system_message, model=model)

def one_chunk_translate_text(model, source_text):
    try:
        source_sentences = nltk.sent_tokenize(source_text)

        st.subheader("Sentence-by-Sentence Translation")
        for sentence in source_sentences:
            translation_1 = one_chunk_initial_translation(model, sentence)
            reflection = one_chunk_reflect_on_translation(model, sentence, translation_1)
            improved_translation = one_chunk_improve_translation(model, sentence, translation_1, reflection)

            st.write(f"{sentence} \n {improved_translation}\n")  # 將原文和譯文一起顯示

        # ... (其他程式碼，例如計算token數量和預估費用) ...

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
    st.subheader("Initial Translation")
    translation_1 = one_chunk_initial_translation(model, source_text)
    st.write(translation_1)

    st.subheader("Translation Reflection")
    reflection = one_chunk_reflect_on_translation(model, source_text, translation_1)
    st.write(reflection)

    st.subheader("Improved Translation")
    improved_translation = one_chunk_improve_translation(model, source_text, translation_1, reflection)
    
    # 分離逐句翻譯和完整翻譯
    sentence_translations, full_translation = improved_translation.split("\n\n", 1)
    
    # 處理逐句翻譯
    pairs = re.split(r'\[SOURCE\]|\[TARGET\]', sentence_translations)
    pairs = [pair.strip() for pair in pairs if pair.strip()]
    
    # 使用 st.table 來展示原文和翻譯
    data = []
    for i in range(0, len(pairs), 2):
        if i+1 < len(pairs):
            data.append({"Original": pairs[i], "Translation": pairs[i+1]})
    
    st.subheader("Sentence-by-Sentence Translation")
    st.table(data)
    
    st.subheader("Full Improved Translation")
    st.write(full_translation)
    
    return full_translation

# Translate button
if st.button("Translate"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not source_text:
        st.error("Please provide some text to translate.")
    else:
        with st.spinner("Translating..."):
            result = one_chunk_translate_text("gpt-4o", source_text)

        # Make sure result is not None before accessing its elements
        if result is not None:
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

    Token Usage:
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
        else:
            st.error("Translation failed. Please check your input and API key.")