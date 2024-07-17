import streamlit as st
import os
from litellm import completion
import PyPDF2
from docx import Document
import tiktoken
import nltk
import ssl
import docx2txt
import io
import zipfile

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
**Modified by:** Tseng Yao Hsien 
**Contact:** zinojeng@gmail.com  
**Reference:** Andrew Ng's AI Agent System for Language Translation [https://github.com/andrewyng/translation-agent](https://github.com/andrewyng/translation-agent)

**Translation Agent** 旨在提升機器翻譯的品質。它採用獨特的「省思式工作流程」，模擬人類翻譯專家的思考過程：

1.初始翻譯: 利用大型語言模型 (LLM) 產生初步譯文。\n
2.反思與改進: 引導 LLM 反思自身譯文，提出改進建議，如同語言審核者般找出不足之處。\n
3.優化輸出: 根據 LLM 的建議，醫療次專科角色，優化譯文，使其更精確、流暢，並符合目標語言的慣用表達。\n
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

# 專科介紹
specialties = {
    "內科": [
        "胸腔內科",
        "職業醫學科",
        "一般內科",
        "內分泌暨新陳代謝科",
        "神經內科",
        "腎臟內科",
        "心臟內科",
        "過敏免疫風濕科",
        "感染科",
        "家庭醫學科",
        "胃腸肝膽科",
        "血液腫瘤科",
    ],
    "外科": [
        "骨科",
        "一般外科",
        "神經外科",
        "泌尿科",
        "胸腔外科",
        "心臟血管外科",
        "整形外科",
        "大腸直腸外科",
        "小兒外科",
    ],
    "其他專科": [
        "婦產部",
        "兒童醫學部",
        "眼科",
        "核子醫學科",
        "麻醉科",
        "耳鼻喉部",
        "皮膚科",
        "心身科",
        "放射腫瘤科",
        "解剖病理科",
        "放射診斷科",
        "臨床病理科",
        "復健醫學部",
        "急診醫學部",
        "口腔醫學部",
        "中醫科",
    ],
}

# 專科選擇
st.subheader("選擇專科 (可選)")
selected_department = st.selectbox("選擇科別 (可選)", ["無"] + list(specialties.keys()))

if selected_department != "無":
    selected_specialty = st.selectbox("選擇專科 (可選)", ["無"] + specialties[selected_department])
else:
    selected_specialty = "無"


# Input method selection
input_method = st.radio("Choose input method:", ("Enter Text", "Upload PDF", "Upload TXT", "Upload Word Document"))

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

# 批量处理部分
st.subheader("Batch Processing")
st.write("Upload multiple files (2 or more) for batch processing.")

uploaded_files = st.file_uploader("Choose files to upload (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) < 2:
        st.warning("Please upload at least 2 files for batch processing.")
    else:
        st.success(f"{len(uploaded_files)} files uploaded successfully.")
        if st.button("Process Batch"):
            with st.spinner("Processing batch files..."):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for uploaded_file in uploaded_files:
                        # 读取文件内容
                        if uploaded_file.name.endswith('.pdf'):
                            content = read_pdf(uploaded_file)
                        elif uploaded_file.name.endswith('.docx'):
                            content = read_doc_or_docx(uploaded_file)
                        elif uploaded_file.name.endswith('.txt'):
                            content = read_txt(uploaded_file)
                        else:
                            st.error(f"Unsupported file format: {uploaded_file.name}")
                            continue
                        
                        # 执行翻译
                        result = one_chunk_translate_text("gpt-4", content)
                        
                        if result:
                            # 创建结果文本
                            result_text = f"""Original Text:
{content}

Improved Translation:
{result['improved_translation']}

Estimated Cost: NTD {result['estimated_cost']:.2f}
"""
                            # 将结果添加到 ZIP 文件
                            zip_file.writestr(f"{uploaded_file.name}_translated.txt", result_text)
                
                # 提供 ZIP 文件下载
                zip_buffer.seek(0)
                st.download_button(
                    label="Download All Translations (ZIP)",
                    data=zip_buffer,
                    file_name="batch_translations.zip",
                    mime="application/zip"
                )


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
        # Add specialty information (if a specialty is selected)
    if selected_specialty != "無":
        prompt += f"This translation is for medical instructions related to {selected_specialty}. Please ensure the terminology is accurate and appropriate for this specialty.\n\n"

    # Complete the prompt
    prompt += """Provide your improved translation as a continuous text, without any additional formatting or labels."""

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