import operator
from typing import Annotated, Dict, TypedDict, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END


llm = ChatOllama(model="qwen3:8b", temperature=0)


class AgentState(TypedDict):
    candidate_url: str
    receiver_url: str
    candidate_raw_text: str
    receiver_raw_text: str
    candidate_info: str  # Structured info extracted by AI
    receiver_info: str   # Structured info extracted by AI
    referral_pitch: str  # The final email


# 3. MOCK SCRAPER FUNCTION

def scrape_profile(url: str, profile_type: str) -> str:
    print(f"--- 🕷️  Scraping {profile_type}: {url} ---")
    
    if "john-doe" in url:
        return """
        Name: John Doe
        Headline: Senior Software Engineer @ UBS
        Location: Zurich, Switzerland
        About: I am a passionate engineer building scalable systems.
        Experience: 
        - Senior Software Engineer at UBS (2020 - Present).
        - Software Developer at Google (2018 - 2020).
        Skills: Python, LangChain, AI Agents, Docker, Kubernetes.
        """
    elif "zach-johnson" in url:
        return """
        Name: Zach Johnson
        Headline: Co-Founder AI Product @ Mindera
        About: Building the future of AI. We are hiring builders!
        Experience: 
        - Co-Founder at Mindera (2021 - Present).
        - Product Lead at Amazon (2016 - 2021).
        """
    else:
        return "Error: Profile content not found."



def get_candidate_profile_content(state: AgentState):
    """Step 1: Get raw text for Candidate"""
    url = state['candidate_url']
    content = scrape_profile(url, "Candidate")
    return {"candidate_raw_text": content}

def get_receiver_profile_content(state: AgentState):
    """Step 2: Get raw text for Receiver"""
    url = state['receiver_url']
    content = scrape_profile(url, "Receiver")
    return {"receiver_raw_text": content}

def extract_candidate_profile_information(state: AgentState):
    """Step 3: Qwen extracts Candidate Info"""
    print("--- 🧠 AI Extracting Candidate Info ---")
    raw_text = state['candidate_raw_text']
    
    prompt = f"""
    Analyze the following LinkedIn profile text. 
    Extract the Name, Current Role, Key Skills, and a brief summary of Experience.
    
    Profile Text:
    {raw_text}
    
    Return the result as a concise summary paragraph.
    """
    response = llm.invoke(prompt)
    return {"candidate_info": response.content}

def extract_receiver_profile_information(state: AgentState):
    """Step 4: Qwen extracts Receiver Info"""
    print("--- 🧠 AI Extracting Receiver Info ---")
    raw_text = state['receiver_raw_text']
    
    prompt = f"""
    Analyze the following LinkedIn profile text.
    Extract the Name, Current Role, and Organization.
    
    Profile Text:
    {raw_text}
    
    Return the result as a concise summary.
    """
    response = llm.invoke(prompt)
    return {"receiver_info": response.content}

def write_a_referral_pitch(state: AgentState):
    """Step 5: Qwen writes the email"""
    print("--- ✍️  AI Writing Referral Pitch ---")
    
    cand_info = state['candidate_info']
    rec_info = state['receiver_info']
    
    prompt = f"""
    You are an expert Career Coach and Copywriter.
    Write a cold outreach message (referral pitch) from the Candidate to the Receiver.
    
    DETAILS:
    - Receiver: {rec_info}
    - Candidate: {cand_info}
    
    INSTRUCTIONS:
    1. Subject Line: Catchy and relevant.
    2. Opening: Mention the receiver's work or company (Mindera).
    3. The Pitch: Connect the candidate's specific skills (Python, AI) to the receiver's company.
    4. Call to Action: Ask for a quick chat or advice on applying.
    5. Tone: Professional but conversational.
    
    Write the email now.
    """
    
    response = llm.invoke(prompt)
    return {"referral_pitch": response.content}



workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("get_candidate_profile_content", get_candidate_profile_content)
workflow.add_node("get_receiver_profile_content", get_receiver_profile_content)
workflow.add_node("extract_candidate_profile_information", extract_candidate_profile_information)
workflow.add_node("extract_receiver_profile_information", extract_receiver_profile_information)
workflow.add_node("write_a_referral_pitch", write_a_referral_pitch)

# Add Logic (Linear Flow)
workflow.set_entry_point("get_candidate_profile_content")
workflow.add_edge("get_candidate_profile_content", "get_receiver_profile_content")
workflow.add_edge("get_receiver_profile_content", "extract_candidate_profile_information")
workflow.add_edge("extract_candidate_profile_information", "extract_receiver_profile_information")
workflow.add_edge("extract_receiver_profile_information", "write_a_referral_pitch")
workflow.add_edge("write_a_referral_pitch", END)


app = workflow.compile()



if __name__ == "__main__":
  
    inputs = {
        "candidate_url": "https://linkedin.com/in/john-doe",
        "receiver_url": "https://linkedin.com/in/zach-johnson"
    }

    print("🚀 STARTING AGENTIC WORKFLOW...\n")
    
   
    result = app.invoke(inputs)

    print("\n" + "="*50)
    print("✅ FINAL OUTPUT: GENERATED REFERRAL PITCH")
    print("="*50 + "\n")
    print(result['referral_pitch'])