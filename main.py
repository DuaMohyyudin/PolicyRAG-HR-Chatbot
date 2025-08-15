import logging
from typing import Dict, Optional
from config import Config
from auth import Authenticator
from data_loader import PolicyDataLoader
from rag_system import PolicyRAGSystem
from query_handler import QueryHandler

# Configure logging
logging.basicConfig(
    filename='hr_assistant.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class HRPolicyAssistant:
    def __init__(self):
        """Initialize the HR Policy Assistant system"""
        print("\n" + "=" * 50)
        print("🚀 Initializing HR Policy Assistant System")
        print("=" * 50 + "\n")

        try:
            # Initialize configuration
            print("🛠 Loading configuration...")
            self.config = Config()
            print("✅ Configuration loaded successfully")

            # Initialize components
            self._initialize_components()

        except Exception as e:
            print(f"❌ System initialization failed: {str(e)}")
            logging.error(f"System initialization failed: {str(e)}")
            raise

    def _initialize_components(self):
        """Initialize all system components"""
        print("\n🔧 Initializing system components...")

        # Initialize Authenticator
        print("🔐 Initializing Authenticator...")
        self.authenticator = Authenticator(self.config)
        try:
            self.authenticator.load_user_data()
            print(f"✅ Authenticator initialized with {len(self.authenticator.user_data)} users")
        except Exception as e:
            print(f"❌ Failed to initialize Authenticator: {str(e)}")
            raise

        # Initialize Policy Data Loader
        print("\n📂 Initializing Policy Data Loader...")
        try:
            self.data_loader = PolicyDataLoader(self.config)
            self.data_loader.load_all_documents()
            print("✅ Policy documents loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load policy documents: {str(e)}")
            raise

        # Initialize RAG System
        print("\n🧠 Initializing RAG System...")
        try:
            self.rag_system = PolicyRAGSystem(
                retriever=self.data_loader.get_retriever(),
                config=self.config,
                data_loader=self.data_loader
            )
            self.rag_system.initialize_llm()
            self.rag_system.setup_qa_chain()
            print("✅ RAG System initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize RAG System: {str(e)}")
            raise

        print("\n✨ All components initialized successfully!")

    def authenticate_user(self) -> Optional[Dict]:
        """Authenticate user with retry logic"""
        max_attempts = self.config.MAX_LOGIN_ATTEMPTS
        print(f"\n🔒 Authentication (Max attempts: {max_attempts})")

        for attempt in range(1, max_attempts + 1):
            try:
                print(f"\nAttempt {attempt}/{max_attempts}")
                username = input("Username: ").strip()
                password = input("Password: ").strip()

                if self.authenticator.authenticate(username, password):
                    user = self.authenticator.get_authenticated_user()
                    print(f"\n✅ Authentication successful! Welcome {user.get('username')}")
                    print(f"👤 User details - Grade: {user.get('grade')}, Gender: {user.get('gender')}")
                    print(f"🍃 Leave Balance: {user.get('remaining_leaves')} days")
                    return user

                print("❌ Invalid credentials. Please try again.")

            except Exception as e:
                print(f"⚠️ Authentication error: {str(e)}")
                logging.error(f"Authentication error: {str(e)}")

        print(f"\n🔐 Maximum login attempts reached. Please contact {self.config.HR_EMAIL}")
        return None

    def run(self):
        """Main execution loop"""
        print("\n" + "=" * 50)
        print("👋 Welcome to HR Policy Assistant")
        print("=" * 50 + "\n")

        try:
            # Authenticate user
            user = self.authenticate_user()
            if not user:
                return

            # Initialize Query Handler with authenticated user
            query_handler = QueryHandler(
                authenticated_user=user,
                rag_system=self.rag_system,
                authenticator=self.authenticator,
                config=self.config
            )

            # Main interaction loop
            print("\n💡 Enter your HR policy queries below (type 'exit' to quit)")
            print("Examples:")
            print("- What is our maternity leave policy?")
            print("- How many leaves do I have remaining?")
            print("- Apply for sick leave 3 days")
            print("- What benefits are available for grade A employees?\n")

            while True:
                try:
                    query = input("\n❓ Your query: ").strip()
                    if query.lower() in ['exit', 'quit']:
                        print("\n👋 Thank you for using HR Policy Assistant. Goodbye!")
                        break

                    if not query:
                        print("⚠️ Please enter a valid query")
                        continue

                    # Process query
                    print("\n🔄 Processing your query...")
                    response = query_handler.handle_query(query)

                    # Display response
                    print("\n" + "=" * 50)
                    print("💬 Response:")
                    print(response)
                    print("=" * 50)

                except KeyboardInterrupt:
                    print("\n👋 Thank you for using HR Policy Assistant. Goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error processing query: {str(e)}")
                    print(f"Please contact {self.config.HR_EMAIL} if the issue persists.")
                    logging.error(f"Query processing error: {str(e)}")

        except Exception as e:
            print(f"\n❌ Fatal error: {str(e)}")
            print(f"Please contact {self.config.HR_EMAIL} for assistance.")
            logging.error(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    try:
        assistant = HRPolicyAssistant()
        assistant.run()
    except Exception as e:
        print(f"\n❌ Critical system error: {str(e)}")
        print("The application will now exit. Please try again later.")
        logging.critical(f"System crash: {str(e)}")
