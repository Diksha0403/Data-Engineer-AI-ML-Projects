FLAN-T5 Summarizer & Q&A Assistant
=====================================
This project provides a command-line AI assistant built using the
google/flan-t5-small model.
It performs two main tasks:

1.  Text Summarization – Summarizes long text into 4–6 bullet points.
2.  Q&A from Local Context – Answers questions using a local context.txt
    file.

------------------------------------------------------------------------

🚀 Features

-   Uses FLAN-T5 Small, a powerful seq2seq model from Google.
-   Summarizes text into concise points.
-   Answers user questions strictly from context.
-   Simple, menu-driven CLI interface.
-   No external API or internet required once the model is downloaded.

------------------------------------------------------------------------

📂 Files in the Project

-   FLAN-T5 Summarizer & Q&A Assistant.py — Main program file.
-   context.txt — Optional file containing custom text for Q&A mode.

------------------------------------------------------------------------

🛠️ Installation & Requirements

Install the required Python libraries:

    pip install transformers torch

(If you are using GPU, install a CUDA-compatible PyTorch version.)

------------------------------------------------------------------------

▶️ How to Use

Run the script:

    python "FLAN-T5 Summarizer & Q&A Assistant.py"

You will see the menu:

    1. Summarize the data
    2. Question & Answer over local context.txt
    0. Exit

1️⃣ Summarization Mode

-   Choose option 1
-   Paste text into the terminal
-   Leave an empty line to end input
-   The model will generate a summary in bullet points

2️⃣ Q&A Mode

-   Create a context.txt file in the same folder
-   Add any text (notes, documents, paragraphs, etc.)
-   Choose option 2
-   Ask a question — the model will answer only from the context

If the answer isn’t in the context, it will respond with “Not found.”

------------------------------------------------------------------------

📘 Example Use Cases

-   Summarize long articles, reports, or study notes
-   Create quick notes from raw text
-   Build a personal assistant for answering questions from
    documentation
-   Use as a command‑line AI tool without internet

------------------------------------------------------------------------

👩‍💻 Author

Diksha Kolikal

