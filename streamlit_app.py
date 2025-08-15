import streamlit as st
import google.generativeai as genai

# Import your existing modules (make sure they're in the same directory)
from config import Config
from auth import Authenticator
from data_loader import PolicyDataLoader
from rag_system import PolicyRAGSystem
from query_handler import QueryHandler

# Page configuration
st.set_page_config(
    page_title="HR Leave Policy Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.375rem;
        border: 1px solid #f5c6cb;
    }
    .leave-balance {
        font-size: 1.5rem;
        font-weight: bold;
        color: #28a745;
        text-align: center;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        border: 2px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitHRAssistant:
    def __init__(self):
        self.config = Config()
        # Configure Gemini API
        genai.configure(api_key=self.config.GEMINI_API_KEY)
        
        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'authenticator' not in st.session_state:
            st.session_state.authenticator = None
        if 'query_handler' not in st.session_state:
            st.session_state.query_handler = None
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = 0

    def initialize_system(self):
        """Initialize the HR system components"""
        if st.session_state.system_initialized:
            return True
            
        try:
            with st.spinner("Initializing HR system..."):
                # Initialize authenticator
                st.session_state.authenticator = Authenticator(self.config)
                
                # Load policy documents
                document_loader = PolicyDataLoader(self.config.POLICY_PDF_PATH)
                document_loader.load_all_documents()
                
                # Initialize RAG system
                rag_system = PolicyRAGSystem(
                    retriever=document_loader.get_retriever(),
                    config=self.config,
                    data_loader=document_loader
                )
                rag_system.initialize_llm()
                rag_system.setup_qa_chain()
                
                st.session_state.rag_system = rag_system
                st.session_state.system_initialized = True
                
                st.success("✅ HR system initialized successfully!")
                return True
                
        except Exception as e:
            st.error(f"❌ Failed to initialize system: {str(e)}")
            return False

    def login_page(self):
        """Display login page"""
        st.markdown("<h1 class='main-header'>🏢 HR Leave Policy Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666;'>GenITeam Solutions</p>", unsafe_allow_html=True)
        
        # Initialize system first
        if not self.initialize_system():
            return
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🔐 Login")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_button = st.form_submit_button("Login", use_container_width=True)
                
                if submit_button:
                    if not username or not password:
                        st.error("Please enter both username and password")
                        return
                    
                    if st.session_state.login_attempts >= self.config.MAX_LOGIN_ATTEMPTS:
                        st.error("Maximum login attempts reached. Please refresh the page to try again.")
                        return
                    
                    try:
                        with st.spinner("Authenticating..."):
                            if st.session_state.authenticator.authenticate(username, password):
                                user = st.session_state.authenticator.get_authenticated_user()
                                st.session_state.user = user
                                st.session_state.authenticated = True
                                st.session_state.login_attempts = 0
                                
                                # Initialize query handler
                                st.session_state.query_handler = QueryHandler(
                                    user, 
                                    st.session_state.rag_system, 
                                    st.session_state.authenticator
                                )
                                
                                st.success(f"✅ Welcome, {username}!")
                                st.rerun()
                            else:
                                st.session_state.login_attempts += 1
                                remaining = self.config.MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                                st.error(f"❌ Invalid credentials. {remaining} attempts remaining.")
                                
                    except Exception as e:
                        st.error(f"⚠️ Authentication error: {str(e)}")
                        st.session_state.login_attempts += 1

    def main_dashboard(self):
        """Display main dashboard after login"""
        # Header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("<h1 class='main-header'>🏢 HR Dashboard</h1>", unsafe_allow_html=True)
        with col2:
            if st.button("Logout", type="secondary"):
                self.logout()
        
        # Sidebar with user info
        with st.sidebar:
            st.markdown("### 👤 User Info")
            st.markdown(f"**Username:** {st.session_state.user['username']}")
            
            # Refresh user data
            if st.button("🔄 Refresh Data", help="Refresh your leave balance"):
                with st.spinner("Refreshing..."):
                    updated_user = st.session_state.authenticator.get_authenticated_user()
                    if updated_user:
                        st.session_state.user = updated_user
                        st.success("Data refreshed!")
                        st.rerun()
            
            # Display current leave balance
            st.markdown("### 📊 Leave Balance")
            leave_balance = st.session_state.user.get('remaining_leaves', 'N/A')
            st.markdown(f"<div class='leave-balance'>{leave_balance} Days</div>", unsafe_allow_html=True)
            
            st.markdown("### ℹ️ Quick Help")
            st.markdown("""
            **What I can help with:**
            - Check leave balance
            - Apply for leave
            - Answer HR policy questions
            - Explain company benefits
            """)
        
        # Main content area
        tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📝 Apply for Leave", "📋 My Info"])
        
        with tab1:
            self.chat_interface()
        
        with tab2:
            self.leave_application_form()
        
        with tab3:
            self.user_info_display()

    def chat_interface(self):
        """Chat interface for policy queries"""
        st.markdown("### 💬 Ask HR Questions")
        st.markdown("Ask me anything about HR policies, benefits, or company procedures!")
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**🧑 You:** {question}")
                st.markdown(f"**🤖 Assistant:** {answer}")
                st.divider()
        
        # Chat input
        with st.form("chat_form"):
            user_query = st.text_area(
                "Your Question:", 
                placeholder="e.g., What is the leave policy? How do I apply for medical reimbursement?",
                height=100
            )
            submit_chat = st.form_submit_button("Ask Question", use_container_width=True)
            
            if submit_chat and user_query:
                with st.spinner("Processing your question..."):
                    try:
                        response = st.session_state.query_handler.handle_query(user_query)
                        st.session_state.chat_history.append((user_query, response))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    def leave_application_form(self):
        """Leave application form"""
        st.markdown("### 📝 Apply for Leave")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.form("leave_form"):
                st.markdown("**Current Leave Balance:** {} days".format(
                    st.session_state.user.get('remaining_leaves', 'N/A')
                ))
                
                leave_days = st.number_input(
                    "Number of Leave Days",
                    min_value=0.5,
                    max_value=float(st.session_state.user.get('remaining_leaves', 30)),
                    step=0.5,
                    value=1.0
                )
                
                leave_reason = st.text_area(
                    "Reason for Leave (Optional)",
                    placeholder="Brief description of why you need leave"
                )
                
                submit_leave = st.form_submit_button("Apply for Leave", use_container_width=True)
                
                if submit_leave:
                    try:
                        with st.spinner("Processing leave application..."):
                            success = st.session_state.authenticator.apply_for_leave(
                                st.session_state.user['username'], 
                                leave_days
                            )
                            
                            if success:
                                # Refresh user data
                                updated_user = st.session_state.authenticator.get_authenticated_user()
                                if updated_user:
                                    st.session_state.user = updated_user
                                
                                st.success(f"""
                                ✅ Leave application submitted successfully!
                                
                                **Details:**
                                - Days applied: {leave_days}
                                - New balance: {st.session_state.user.get('remaining_leaves', 'N/A')} days
                                - Status: Pending approval
                                """)
                                
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
        
        with col2:
            st.markdown("#### 📋 Leave Policy Summary")
            st.info("""
            **Leave Rules:**
            - Minimum: 0.5 days
            - Maximum: 30 days per application
            - Must have sufficient balance
            - Requires manager approval
            """)

    def user_info_display(self):
        """Display user information"""
        st.markdown("### 📋 My Information")
        
        user_data = st.session_state.user
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Personal Details")
            st.text(f"Username: {user_data.get('username', 'N/A')}")
            
        with col2:
            st.markdown("#### Leave Information")
            st.text(f"Remaining Leaves: {user_data.get('remaining_leaves', 'N/A')} days")
        
        # Display all user data in an expandable section
        with st.expander("View All Details"):
            st.json(user_data)

    def logout(self):
        """Handle user logout"""
        # Clear session state
        for key in list(st.session_state.keys()):
            if key not in ['system_initialized', 'rag_system', 'authenticator']:
                del st.session_state[key]
        
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.query_handler = None
        st.session_state.login_attempts = 0
        
        st.success("👋 Logged out successfully!")
        st.rerun()

    def run(self):
        """Main application runner"""
        if not st.session_state.authenticated:
            self.login_page()
        else:
            self.main_dashboard()

# Main execution
if __name__ == "__main__":
    app = StreamlitHRAssistant()
    app.run()