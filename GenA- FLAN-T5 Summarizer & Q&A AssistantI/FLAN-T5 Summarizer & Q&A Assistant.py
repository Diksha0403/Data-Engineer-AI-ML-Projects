#################################################################### 
# FLAN-T5 Model 
####################################################################
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import from Hugging Face Transformers 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# AutoTokenizer : A tokenizer loader that automatically picks the right tokenizer for the model you choose. 
# AutoModelForSeq2SeqLM : A pre-trained model loader for Sequence-to-Sequence Language Models (Seq2SeqLM).

# Choose a  instruction-tuned model 
MODEL_NAME = "google/flan-t5-small" 
# A lightweight version of FLAN-T5. 
# About 80 million parameters. 

print(" FLAN-T5_Summarizer_Q&A_Assistant {MODEL_NAME} model loading...")

# Load tokenizer (handles text <-> tokens). 
# AutoTokenizer picks the right tokenizer for the model.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load the sequence-to-sequence model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

####################################################################
# This function takes a text prompt (string) uses a FLAN-T5 model to generate a  
# continuation/answer and returns the generated text.  
####################################################################

def run_flan(prompt: str, max_new_tokens:int = 128)-> str:
    # Tokenisation 
    # Tokenize the input prompt; return PyTorch tensors; truncate if too long
    inputs = tokenizer(prompt,return_tensors="pt",truncation=True)

    # Generation 
    # Generate text from the model with light sampling for naturalness 
    outputs = model.generate(
        **inputs,                            # pass tokenized inputs (input_ids, attention_mask)
        max_new_tokens=max_new_tokens,        # how many new tokens to generate
        do_sample=True,                       # enable random sampling 
        top_p=0.9,                            # nucleus sampling: only consider tokens in the top 90% probability mass 
        temperature=0.7                       # control randomness (lower = safer/more deterministic)
    )
    # Decode token IDs back into a clean string 
    # Example IDs: [71, 867, 1234, 42, 1] 
    # Text: "Hello, how are you?"
    return tokenizer.decode(outputs[0],skip_special_tokens=True).strip()

####################################################################
#   This functoin is used for summarisation                                       
#   It creates a prompt with 4 to 6 bullet points 
####################################################################
def summarize_text(text:str)-> str:
    # Prompt template instructing the model to produce 4–6 bullet points
    prompt = f"Summarize the following text in 4-6 bullet points:\n\n{text}"

    # Allow a slightly longer output for bullet lists
    return run_flan(prompt,max_new_tokens=160)
    
####################################################################
#   This functoin is used to laod the contents form our local file 
#   And return the complete file contents in one string 
####################################################################

def load_context(path: str = "context.txt")-> str:
    try:
        # Read the entire file as a single string 
        with open(path,"r",encoding="utf-8")as f:
            return f.read()
    except FileNotFoundError:
        return ""

####################################################################
#   This functoin ask FLAN to answer using ONLY the given context. 
#   If answer isn't present, ask it to say 'Not found'.                                
####################################################################

def answer_from_context(question:str,context:str)-> str:
    if not context.strip():
        return "Context file not found or empty.Create 'context.txt' first"
    # Construct a strict prompt for FLAN-T5
    prompt = (
        "You are a helpful assistant.Answer the question ONLY using the context.\n"
        "If the answer is not in the context, reply exactly: Not found.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:{question}\nAnswer:"
    )
    # Generate a concise answer grounded in the provided notes
    return run_flan(prompt,max_new_tokens=120)
    
####################################################################
#   Entry point function
####################################################################

def main():
    print("------------------------------------")
    print("\n------------Welcome in FLAN-T5 Model-------------")
    print("1. Summarize the data")
    print("2. Question & Answer over local context.txt")
    print("0. Exit")
    print("------------------------------------")

    while True:
        choice = input("\n Choose an option(1/2/0):").strip()
    
        if choice == "0":
            print("Thank you for using FLAN-T5 Model")
            break
            
        elif choice == "1":
             # Collect multiple lines of text for summarization
            print("You have selected Summarisation option...")
            print("\n Paste text to summarize. ENd with a blank line:")

            lines = []
            while True:
                line = input()
                
                # Stop when the user hits Enter on an empty line
                if not line.strip():
                    break  
                lines.append(line)

             # Join lines into a single block of text 
            text = "\n".join(lines).strip()
            
             # If no text was provided, prompt again 
            if not text:
                print("FLAN-T5 says: No text received.")
                continue

            # Run summarization and print the result
            print("\n Summary generated by FLAN model:")
            print(summarize_text(text))

        elif choice == "2":
             # Load context from local file 'context.txt'
            ctx = load_context("context.txt")
        
            if not ctx.strip():
                # Help the user if the context is missing/empty
                print("Missing 'context.txt'. Create it in the same folder and try again.")
                continue

             # Ask a question related to the provided context
            q = input("\n Ask a question about your context to FLAN model:").strip()
            if not q:
                print("No question received.")
                continue

            # Generate an answer grounded only in the context
            print("\n Answer from FLAN model:")
            print(answer_from_context(q,ctx))
        
        else:
            print("Please choose 1,2, or 0")

###################################################################
#   Starter 
###################################################################

if __name__== "__main__": 

    main()
