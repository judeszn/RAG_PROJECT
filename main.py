import os
import base64
import json
import uuid
import datetime
from io import BytesIO
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from PIL import Image
import gradio as gr
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel, Field
from dotenv import load_dotenv

# --- Environment Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"
app = FastAPI(
    title="Multimodal AI Assistant",
    description="Chatbot with PDF parsing, image understanding, and multimodal capabilities",
    version="2.0"
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- File Management ---
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Gemini Initialization ---
genai.configure(api_key=GEMINI_API_KEY)
text_model = genai.GenerativeModel(GEMINI_MODEL)
vision_model = genai.GenerativeModel("gemini-2.0-flash")

# --- Pydantic Models for Structured Output ---
class ResumeData(BaseModel):
    name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address")
    phone: Optional[str] = Field(description="Phone number")
    summary: str = Field(description="Professional summary")
    experience: List[Dict] = Field(description="List of work experiences")
    education: List[Dict] = Field(description="List of educational qualifications")
    skills: List[str] = Field(description="List of skills")
    certifications: Optional[List[str]] = Field(description="List of certifications")

# --- Core Functionality ---
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF with error handling"""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        raise HTTPException(500, f"PDF processing error: {str(e)}")

async def save_file(file: UploadFile) -> str:
    """Save uploaded file with unique filename"""
    file_ext = os.path.splitext(file.filename)[1]
    file_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}{file_ext}"
    file_path = os.path.join("uploads", file_id)
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    
    return file_path

def generate_image_description(image: Image.Image, prompt: str) -> str:
    """Generate description for an image"""
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    response = vision_model.generate_content([prompt, image])
    return response.text

async def parse_resume_text(resume_text: str) -> Dict:
    """Parse resume text using structured output"""
    parser = JsonOutputParser(pydantic_object=ResumeData)
    prompt = ChatPromptTemplate.from_template(
        "Parse this resume into structured JSON:\n{resume_text}\n{format_instructions}"
    )
    chain = prompt | ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.1, google_api_key=GEMINI_API_KEY).with_config({"run_name": "LangSmith tracing"}) | parser
    return await chain.ainvoke({"resume_text": resume_text, "format_instructions": parser.get_format_instructions()})

async def chat_with_history(message: str, history: list, image: Optional[Image.Image] = None) -> str:
    """Chat with context awareness and image understanding"""
    chat = text_model.start_chat(history=history)
    
    if image:
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        response = await chat.send_message_async([
            {"text": message },
            {"inline_data": {"mime_type": "image/jpeg", "data": image_data}}
        ])
    else:
        response = await chat.send_message_async(message)
    
    return response.text

# --- API Endpoint ---
@app.post("/api/parse_resume/")
async def parse_resume(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are supported")
    try:
        pdf_bytes = await file.read()
        resume_text = extract_text_from_pdf(pdf_bytes)  # No await here!
        parsed_data = await parse_resume_text(resume_text)
        result_id = f"resume_{uuid.uuid4().hex}.json"
        result_path = os.path.join("processed", result_id)
        with open(result_path, "w") as f:
            json.dump(parsed_data, f, indent=2)
        return {
            "status": "success",
            "result_id": result_id,
            "data": parsed_data
        }
    except Exception as e:
        print(f"Error in /api/parse_resume/: {e}")  # Add this line for debugging
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.post("/api/describe_image")
async def describe_image(
    file: UploadFile = File(...),
    prompt: str = Form("Describe this image in detail")
):
    """Generate description for an uploaded image"""
    try:
        image = Image.open(BytesIO(await file.read()))
        description = generate_image_description(image, prompt)
        return {"description": description}
    except Exception as e:
        raise HTTPException(500, f"Image processing error: {str(e)}")

@app.post("/api/chat")
async def chat_endpoint(
    message: str = Form(...),
    image: Optional[UploadFile] = File(None),
    history: str = Form("[]")
):
    """ Chat endpoint with image support """
    try:
        history_data = json.loads(history)
        img_obj = Image.open(BytesIO(await image.read())) if image else None

        response = await chat_with_history(message, history_data, img_obj)
        return {"response": response}
    except Exception as e:
        raise HTTPException(500, f"Chat error: {str(e)}")

@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    """Retrieve processed results"""
    file_path = os.path.join("processed", result_id)
    if not os.path.exists(file_path):
        raise HTTPException(404, "Result not found")
    return FileResponse(file_path)

# --- Gradio Interface ---
async def gradio_parse_resume(pdf_file):
    """Gradio interface for resume parsing"""
    with open(pdf_file.name, "rb") as f:
        resume_text = extract_text_from_pdf(f.read())
    
    parsed_data = await parse_resume_text(resume_text)
    return parsed_data

async def gradio_image_describer(image, prompt):
    """Gradio interface for image description"""
    return generate_image_description(image, prompt)

async def gradio_chat_interface(message, history, image):
    """Gradio chat interface with image support"""
    img_obj = image if image is not None else None
    history_list = [(h[0], h[1]) for h in history]  # Convert to list of tuples
    
    try:
        response = await chat_with_history(message, history_list, img_obj)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# --- Enhanced UI with Tabs ---
with gr.Blocks(title="Multimodal AI Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("#  Multimodal AI Assistant")
    gr.Markdown("Upload documents, chat with AI, or analyze images")
    
    with gr.Tab(" Chatbot"):
        gr.Markdown("Chat with the AI assistant using text.")
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(label="Type your message...", scale=4)
        submit_btn = gr.Button("Send", scale=0)
        clear_btn = gr.Button("Clear Chat")

        async def respond(message, chat_history):
            if not message.strip():
                return "", chat_history
            response = await gradio_chat_interface(message, chat_history, None)
            chat_history = chat_history + [(message, response)]
            return "", chat_history

        submit_btn.click(
            respond,
            [msg, chatbot],
            [msg, chatbot],
            queue=True
        )
        msg.submit(
            respond,
            [msg, chatbot],
            [msg, chatbot],
            queue=True
        )
        clear_btn.click(lambda: None, None, chatbot, queue=False)

    with gr.Tab(" Resume Parser"):
        gr.Markdown("Upload a PDF resume to extract structured data")
        pdf_input = gr.File(label="PDF Resume", file_types=[".pdf"])
        parse_btn = gr.Button("Parse Resume")
        json_output = gr.JSON(label="Parsed Data")
        parse_btn.click(gradio_parse_resume, pdf_input, json_output)

    with gr.Tab(" Image Analyzer"):
        gr.Markdown("Upload an image and get AI-powered description")
        img_analyzer = gr.Image(label="Upload Image", type="pil")
        prompt_input = gr.Textbox(label="Prompt", value="Describe this image in detail")
        analyze_btn = gr.Button("Analyze Image")
        analysis_output = gr.Textbox(label="Description", interactive=False)
        analyze_btn.click(gradio_image_describer, [img_analyzer, prompt_input], analysis_output)

# --- Mount Gradio to FastAPI ---
app = gr.mount_gradio_app(app, demo, path="")

# --- Frontend Integration ---
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
    <html>
        <head>
            <title>Multimodal AI Assistant</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
                .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
                header { background: #4f46e5; color: white; padding: 20px; text-align: center; }
                .features { display: flex; flex-wrap: wrap; gap: 20px; margin: 30px 0; }
                .card { background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
                        padding: 20px; flex: 1; min-width: 300px; }
                .btn { display: block; background: #4f46e5; color: white; text-align: center; 
                      padding: 15px; border-radius: 5px; text-decoration: none; font-weight: bold; margin-top: 15px; }
                iframe { width: 100%; height: 600px; border: none; border-radius: 10px; }
            </style>
        </head>
        <body>
            <header>
                <h1>Multimodal AI Assistant</h1>
                <p>Runs pdf documents, analyze images, and extract insights</p>
            </header>
            <div class="container">
                <div class="features">
                    <div class="card">
                        <h2>AI Chatbot</h2>
                        <p>Chat with our AI assistant using text and images</p>
                        <a href="/" class="btn">Open Chat</a>
                    </div>
                    <div class="card">
                        <h2>Resume Parser</h2>
                        <p>Upload PDF resumes to extract structured data</p>
                        <a href="/#tab=📄%20Resume%20Parser" class="btn">Parse Resumes</a>
                    </div>
                    <div class="card">
                        <h2>Image Analysis</h2>
                        <p>Get detailed descriptions of uploaded images</p>
                        <a href="/#tab=🖼%20Image%20Analyzer" class="btn">Analyze Images</a>
                    </div>
                </div>
                <iframe src="/"></iframe>
            </div>
        </body>

    </html>
    """

# --- Run Application ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)