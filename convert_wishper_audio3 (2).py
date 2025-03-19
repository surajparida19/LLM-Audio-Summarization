import os
import requests
from io import BytesIO
import logging
from datetime import datetime
import whisper
import librosa  # Ensure this is installed

from db_helpers.connection_helpers import get_database_connection
from s3_helpers.s3_helpers import S3Uploader

##Importing Environment Variables
from dotenv import load_dotenv

##Importing Gemini
import google.generativeai as genai

##Importing LangChain Libraries
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


##=======================================================================================================
##Initializations
##=======================================================================================================
##Loading all the environment variables from .env file
load_dotenv() 

##Configuring our Gemini API
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

##Setting up the LLM - NEEDS CHANGE ONCE MODEL IS DECIDED
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

##Setting up the LangSmith Tracking Capabilities
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")


##=======================================================================================================
##Function Definitions
##=======================================================================================================
class FileExtractService:
    @staticmethod
    def fetch_file_from_url(url):
        """
        Fetch file content from a URL and return a BytesIO object along with its content type.
        """
        response = requests.get(url)
        response.raise_for_status()  # Raise error if the request fails
        return BytesIO(response.content), response.headers.get("Content-Type", "").lower()

def process_audio_in_memory(s3_url, model):
    """
    Uses FileExtractService to fetch the audio file as a BytesIO object,
    loads it into a numpy array using librosa, and transcribes it with Whisper.
    
    Supports mp3, wav, and m4a formats.
    Returns the transcription text.
    """
    try:
        audio_io, content_type = FileExtractService.fetch_file_from_url(s3_url)
        audio_io.seek(0)  # Ensure the BytesIO pointer is at the start

        # Assign a name attribute based on the content type to help librosa detect the format.
        if "mp3" in content_type:
            audio_io.name = "temp.mp3"
        elif "wav" in content_type:
            audio_io.name = "temp.wav"
        elif "m4a" in content_type or "mp4" in content_type or "aac" in content_type:
            audio_io.name = "temp.m4a"
        else:
            # Default to mp3 if not explicitly provided
            audio_io.name = "temp.mp3"

        # Load the audio from the in-memory BytesIO object (with desired sample rate, e.g., 16000 Hz)
        audio, sr = librosa.load(audio_io, sr=16000)
        # Transcribe the audio (Whisper accepts a numpy array)
        result = model.transcribe(audio)
        print(f"result---{result}",flush=1)
        transcription = result["text"]

        ##LLM Layer for creating summary
        transcription_summary_prompt = ("""You are a professional summarization AI. Your task is to analyze a given transcript and generate a **concise, professional summary** in **markdown format** along with a file name derived from the summary title.

        ### **Instructions:**  
        - **Title:** Create a clear, informative title for the summary.  
        - **File Name:** Generate a file name based on the summary title. The file name should be all lowercase, use underscores instead of spaces).  
        - **Summary:** Provide a structured summary that captures key points.  
        - Use **headings, bullet points, or numbered lists** if the transcript is long or complex to improve readability.  
        - **Do not include** filler phrases like “Here is the summary” or any introductory/explanatory text.  
        - **Output strictly in markdown format.**

        ### **Output Format:**
        File Name: [Generated File Name]
        # Summary: [Title]  
        [Summary]


        ### **Example Input**
        [Full transcript of a conversation, meeting, interview, etc.]

        ### **Example Output***
        File Name: team_strategy_meeting_q2_goals.md  
        # Summary: Team Strategy Meeting - Q2 Goals  

        ## Key Discussion Points  
        - **Sales Performance:** The team reviewed Q1 numbers, showing a **15 percent increase in revenue**.  
        - **Marketing Strategy:** Focus on **social media outreach** and **email campaigns** to improve engagement.  
        - **Product Development:** New feature rollout scheduled for **May 2024**.  
        - **Action Items:**  
        1. Finalize Q2 marketing budget.  
        2. Conduct user feedback survey.  
        3. Plan a product launch webinar.


        Here is input the transcript:
        {user_transcript} 
        """)

         ##Creating the actual prompt which the LLM will use
        transcription_summary_prompt_template = ChatPromptTemplate.from_template(transcription_summary_prompt)

        ##Creating a basic chain for the prompt
        transcription_summary_generation = transcription_summary_prompt_template | llm | StrOutputParser()

        # running the chain
        response = transcription_summary_generation.invoke({"user_transcript": transcription})

        # Retrieve the file name and remove it from the generated summary
        lines = response.splitlines()
        file_name = ""
        if lines and lines[0].startswith("File Name:"):
            file_name = lines[0].replace("File Name:", "").strip()
            # Remove the file name line from the response
            summary_without_filename = "\n".join(lines[1:]).strip()
        else:
            summary_without_filename = response

        # Append the original transcript without the file name
        final_response = summary_without_filename + "\n" + "# Transcript:" + "\n" + transcription

        logging.info("Transcription: " + final_response)
        print(final_response, flush=1)
        # Return both the extracted file name and the final response
        return file_name, final_response
    except Exception as e:
        logging.error("Error processing audio in memory: " + str(e))
        return None

def create_transcription_file_io(transcription):
    """
    Converts transcription text into a BytesIO object.
    """
    return BytesIO(transcription.encode('utf-8'))

def call_space_generation_api(new_txt_url, file_name, uid):
    """
    Calls the text_to_space API to generate a space from text.
    Returns the parsed JSON response.
    """
    api_url = "https://testbackend.askpresto.com/spaces/text_to_space/"
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
        'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjhPMU5HeUp1cUxBQWNta25ZeUpHUiJ9.eyJnaXZlbl9uYW1lIjoiSmF5ZXNoIEt1bWFyIiwiZmFtaWx5X25hbWUiOiJLZXNocmkiLCJuaWNrbmFtZSI6ImpheWVzaC5rZXNocmkiLCJuYW1lIjoiSmF5ZXNoIEt1bWFyIEtlc2hyaSIsInBpY3R1cmUiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NKNVZnNHl1TW5Ua25rMkhuQ3MxUU5BRlpBLUxvc1ptZkRyNlpXd2U0a2ZkMFRGLXc9czk2LWMiLCJ1cGRhdGVkX2F0IjoiMjAyNS0wMi0yNVQwODozMzoyOC42NTJaIiwiZW1haWwiOiJqYXllc2gua2VzaHJpQHByZXN0b2xhYnMuYWkiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaXNzIjoiaHR0cHM6Ly9wcmVzdG8udWsuYXV0aDAuY29tLyIsImF1ZCI6IkhJa1Y0RjkwaGJqMEs3UFQ3Z0dpdFQzZzRhNFE3U2ZDIiwic3ViIjoiZ29vZ2xlLW9hdXRoMnwxMDk1NTA1MDEwOTUxOTQ2MzY4NDQiLCJpYXQiOjE3NDA0NzI0MTEsImV4cCI6MTc0MDUwODQxMSwic2lkIjoieS1lRm5GVTFydldUQ3FyVEU2YlpaSzd3d2tHTHNYeUMiLCJub25jZSI6Ik1FZFpjVU5LZVZWdVdrcHdOR1JsTlZGUlVISklYMmxWTXpGMGFVMUxhazlmY0U1eU1IVklXbE5KYUE9PSJ9.pQYMOdkXsPQjSKOoHXzQy71FYz03T-vf49XfWH-RCkNLRvTVNy4pf_lqiT85C48HSLGoqa0qIdNf6b83H33tOnXptmo9UMY4vrjVXZg-LyB4MxSnOVKpbUM0k5KWtAWiOgsZA1BLvDhgyHVxK67vI2JjpzxcWC6xfxdeO37vVGYMVL1C1VqSpc_5SgIIhyxgKp0RynUmM3aRhVYdW8mGlIYRXRANvyIAYYwS0DH3La6Cd2Y54wprVirdPg3-66m8nsR-MTqC57jDkdCzJ8eS7dZXVb86l0cMyJgy0k6ysVw5Fyj4nz_E6XgeXSpvyCXPsGFyifgWeeNXeq5Pj2teiw',
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        'Origin': 'https://test.permi.tech',
        'Referer': 'https://test.permi.tech/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'cross-site',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"'
    }
    payload = {
        "file_name": file_name,
        "url": new_txt_url,
        "workspace_id": "5c28596f-9791-4fa0-b9ff-53b7e01170b2",
        "org_id": "0a8e66f1-faad-da5c-d8c6-b6dbd69ca945",
        "uid": uid,
        "board_id": "ff127726-616b-4df6-b76f-84acbc79fe11"
    }
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def update_record(record_id, new_txt_url, document_id, conn):
    """
    Updates the database record with the new txt_file_url, document_id,
    marks the status as 'complete', and updates the timestamp.
    """
    try:
        with conn.cursor() as cursor:
            update_query = """
                UPDATE audio_conversion_data
                SET txt_file_url = %s,
                    document_id = %s,
                    status = 'complete',
                    updated_at = %s
                WHERE id = %s
            """
            updated_at = datetime.utcnow()
            cursor.execute(update_query, (new_txt_url, document_id, updated_at, record_id))
            conn.commit()
            logging.info(f"Record {record_id} updated with txt_file_url: {new_txt_url} and document_id: {document_id}")
    except Exception as e:
        logging.error(f"Failed to update record {record_id}: " + str(e))
        conn.rollback()

def main():
    logging.basicConfig(level=logging.INFO)
    
    # Establish DB connection and load the Whisper model
    conn = get_database_connection()
    model = whisper.load_model("base")
    print(f"mode")
    s3_uploader = S3Uploader()
    
    while True:
        with conn.cursor() as cursor:
            fetch_query = """
                SELECT id, audio_file_url, created_by
                FROM audio_conversion_data
                WHERE status = 'pending'
                LIMIT 10;
            """
            cursor.execute(fetch_query)
            records = cursor.fetchall()
            print(f"records------{records}",flush=1) 
        if not records:
            logging.info("No pending records found. Exiting batch processing.")
            print("No pending records found. Exiting batch processing.",flush=1)
            break
        
        for record in records:
            record_id, audio_file_url, uid = record
            logging.info(f"Processing record: {record_id}")
            
            # Process the audio in memory and get the transcription.
            file_name, transcription = process_audio_in_memory(audio_file_url, model)
            if not transcription:
                logging.error(f"Skipping record {record_id} due to processing error.")
                continue
            
            # Create an in-memory file for the transcription text.
            transcription_file_io = create_transcription_file_io(transcription)
            filename = f"{filename}.txt"
            file_slug = f"{filename}"
            
            try:
                # Upload the transcription file to S3.
                new_txt_url = s3_uploader.upload_file(transcription_file_io, filename, file_slug)
                logging.info(f"Uploaded transcription file to: {new_txt_url}")
            except Exception as e:
                logging.error(f"Failed to upload transcription file for record {record_id}: {e}")
                continue
            
            try:
                # Call the space generation API to create a space from the text.
                api_response = call_space_generation_api(new_txt_url, filename, uid)
                document_id = api_response.get("document", {}).get("id")
                if not document_id:
                    logging.error(f"API response did not contain a document id for record {record_id}.")
                    continue
            except Exception as e:
                logging.error(f"Error calling space generation API for record {record_id}: {e}")
                continue
            
            # Update the DB record with the new transcription file URL and document id.
            update_record(record_id, new_txt_url, document_id, conn)
            logging.info(f"Record {record_id} processed and updated successfully.")
    
    conn.close()

if __name__ == "__main__":
    main()

