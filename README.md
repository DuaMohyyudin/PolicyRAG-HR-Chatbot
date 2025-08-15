HR Policy Assistant System - Comprehensive Documentation
=====================================================

Overview
--------
The HR Policy Assistant is an intelligent system designed to help employees and HR personnel quickly access and understand company policies, particularly regarding leave management and benefits. The system combines document retrieval, natural language processing, and a conversational interface to provide accurate, context-aware responses to policy-related queries.

System Architecture
-------------------
The system follows a modular architecture with these main components:

1. Authentication Module - Handles user authentication against Google Sheets data
2. Data Loading Module - Processes PDF policies and Excel/Google Sheets data
3. RAG System - Implements document retrieval and response generation
4. Query Handler - Processes user queries and routes them appropriately
5. Main Application - Coordinates all components and handles user interaction

Core Technologies
----------------
- Python 3.9+
- Google Gemini API
- Google Sheets API
- FAISS (Facebook AI Similarity Search)
- Sentence Transformers
- LangChain
- Pandas
- scikit-learn

File Structure and Code Flow
---------------------------

1. auth.py - Authentication Module
   - Purpose: Handles user authentication and leave management
   - Key Components: Authenticator class, Google Sheets integration
   - Main Methods: _connect_to_google_sheets(), load_user_data(), authenticate(), apply_for_leave()

2. data_loader.py - Data Loading Module
   - Purpose: Loads and processes policy documents and user data
   - Key Components: PolicyDataLoader class, hybrid document processing
   - Main Methods: _load_pdf_documents(), _load_excel_documents(), load_all_documents()

3. rag_system.py - RAG Implementation
   - Purpose: Implements retrieval-augmented generation for policy queries
   - Key Components: PolicyRAGSystem class, hybrid search
   - Main Methods: initialize_llm(), _hybrid_search(), query_policy()

4. query_handler.py - Query Processing
   - Purpose: Routes and processes different types of user queries
   - Key Components: QueryHandler class, specialized query handlers
   - Main Methods: _handle_leave_application(), handle_query()

5. main.py - Application Entry Point
   - Purpose: Coordinates all system components
   - Key Components: HRPolicyAssistant class
   - Main Methods: _initialize_components(), authenticate_user(), run()

Key Features
------------
- Real-time authentication against Google Sheets
- Hybrid document processing (PDF + Excel/Sheets)
- Context-aware policy responses
- Leave management system
- Grade/gender-specific answer generation
- Comprehensive error handling

Setup Instructions
-----------------
1. Install requirements: pip install -r requirements.txt
2. Set up Google Sheets with user data
3. Configure Gemini API key in config.py
4. Place policy PDF in specified location
5. Run application: python main.py

Usage Examples
-------------
1. Authentication:
   Username: john_doe
   Password: 1234

2. Policy Queries:
   What is our maternity leave policy?
   How many leaves do I have remaining?

3. Leave Applications:
   apply for sick leave 3 days

Configuration
-------------
Edit config.py to set:
- Google Sheets ID and credentials
- Gemini API key
- Policy document paths
- Application settings (MAX_LOGIN_ATTEMPTS, etc.)

Error Handling
-------------
- Comprehensive logging to hr_assistant.log
- User-friendly error messages
- Multiple fallback mechanisms
- Automatic data refresh on errors

Limitations
----------
- Best for structured policy documents
- Google Sheets backend may not scale for large organizations
- Requires well-formatted input queries


Security Notes
-------------
- All Google API communication is encrypted
- Passwords stored as integers only
- Service account credentials are file-based
- No sensitive data in logs

