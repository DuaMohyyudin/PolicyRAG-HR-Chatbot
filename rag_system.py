import logging
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from io import StringIO

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

class PolicyRAGSystem:
    def __init__(self, retriever, config, data_loader=None):
        """Initialize the RAG system with retriever and configuration"""
        print("🔄 Initializing PolicyRAGSystem...")
        self.retriever = retriever
        self.data_loader = data_loader
        self.llm = None
        self.chain = None
        self.config = config
        self.llm_ready = False
        self.llm_type = None
        self.embedder = SentenceTransformer(self.config.EMBEDDING_MODEL)
        
        # Store user data from Excel
        self.user_database = {}
        self._load_user_data()
        
        logging.basicConfig(
            filename='rag_logs.txt',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        print("🔑 Configuring AI API...")

    def _load_user_data(self):
        """Load user data from Excel sheets in data_loader"""
        print("📊 Loading user database from Excel data...")
        try:
            if not self.data_loader or not hasattr(self.data_loader, 'all_chunks'):
                print("⚠️ No data loader or chunks available")
                return
            
            # Find Excel documents in the loaded chunks
            for doc in self.data_loader.all_chunks:
                if doc.metadata.get('source', '').startswith('sheet:'):
                    sheet_name = doc.metadata['source'].replace('sheet:', '')
                    print(f"🔍 Processing sheet: {sheet_name}")
                    
                    # Try to parse as CSV first, then fallback to other methods
                    try:
                        # Clean the content by removing extra whitespace
                        clean_content = '\n'.join([line.strip() for line in doc.page_content.split('\n') if line.strip()])
                        
                        # First try with CSV parsing
                        try:
                            df = pd.read_csv(StringIO(clean_content))
                        except:
                            # If CSV fails, try with space separator
                            df = pd.read_csv(StringIO(clean_content), sep='\s+', engine='python')
                        
                        # Normalize column names (case insensitive)
                        df.columns = [col.strip().lower() for col in df.columns]
                        
                        # Check if this is a user data sheet
                        if 'username' in df.columns and 'grade' in df.columns:
                            for _, row in df.iterrows():
                                user_data = {
                                    'username': str(row.get('username', '')).lower().strip(),
                                    'grade': str(row.get('grade', 'Unknown')).strip(),
                                    'gender': str(row.get('gender', row.get('sex', 'not specified'))).lower().strip(),
                                    'remaining_leaves': str(row.get('remaining_leaves', row.get('leaves', 'N/A'))),
                                    'total_leaves': str(row.get('total_leaves', 'N/A'))
                                }
                                if user_data['username']:
                                    self.user_database[user_data['username']] = user_data
                                    print(f"👤 Added user: {user_data['username']} (Grade: {user_data['grade']}, Gender: {user_data['gender']})")
                    except Exception as e:
                        print(f"⚠️ Error parsing sheet {sheet_name}: {str(e)}")
                        continue
            
            print(f"✅ User database loaded with {len(self.user_database)} users")
            
        except Exception as e:
            print(f"❌ Error loading user data: {str(e)}")
            logging.error(f"User data loading failed: {str(e)}")

    def get_user_data(self, username: str) -> Dict:
        """Get user data by username with case-insensitive lookup"""
        return self.user_database.get(username.lower().strip(), {})

    def initialize_llm(self) -> None:
        """Initialize the Gemini model with LangChain compatibility"""
        print("\n🔧 Initializing Gemini LLM...")
        try:
            # Try LangChain's GoogleGenerativeAI first (recommended)
            try:
                from langchain_google_genai import GoogleGenerativeAI
                
                self.llm = GoogleGenerativeAI(
                    model=self.config.GEMINI_MODEL,
                    google_api_key=self.config.GEMINI_API_KEY,
                    temperature=self.config.GEMINI_TEMPERATURE,
                    max_output_tokens=2048,
                    top_p=0.95,
                    top_k=40
                )
                
                self.llm_ready = True
                self.llm_type = "langchain"
                logging.info(f"LangChain GoogleGenerativeAI {self.config.GEMINI_MODEL} initialized successfully")
                print(f"✅ LangChain GoogleGenerativeAI {self.config.GEMINI_MODEL} initialized successfully")
                
            except ImportError as import_error:
                print(f"⚠️ LangChain Google GenAI not available: {import_error}")
                print("🔄 Falling back to Google AI SDK...")
                
                import google.generativeai as genai
                genai.configure(api_key=self.config.GEMINI_API_KEY)
                
                self.llm = genai.GenerativeModel(
                    model_name=self.config.GEMINI_MODEL,
                    generation_config={
                        "temperature": self.config.GEMINI_TEMPERATURE,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 2048
                    },
                    safety_settings={
                        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
                    }
                )
                self.llm_ready = True
                self.llm_type = "google_ai"
                print(f"✅ Google AI SDK {self.config.GEMINI_MODEL} initialized successfully")
                
        except Exception as e:
            logging.error(f"Failed to initialize Gemini: {str(e)}")
            print(f"❌ Failed to initialize Gemini: {str(e)}")
            raise Exception(f"LLM initialization failed: {str(e)}")

    def _semantic_search(self, query: str, chunks: List[str], top_k: int = 5) -> List[tuple]:
        """Perform semantic search using cosine similarity"""
        print(f"🧠 Performing semantic search for: '{query}'")
        try:
            query_embedding = self.embedder.encode([query])
            chunk_embeddings = self.embedder.encode(chunks)
            
            similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
            
            indexed_similarities = [(sim, i, chunks[i]) for i, sim in enumerate(similarities)]
            indexed_similarities.sort(key=lambda x: x[0], reverse=True)
            
            print(f"🎯 Semantic search completed, top similarity: {indexed_similarities[0][0]:.4f}")
            return indexed_similarities[:top_k]
            
        except Exception as e:
            print(f"❌ Semantic search failed: {str(e)}")
            return []

    def _is_insurance_query(self, query: str) -> bool:
        """Check if query is about insurance/health benefits"""
        insurance_keywords = [
            'insurance', 'health', 'medical', 'benefits', 'coverage',
            'hospital', 'opd', 'doctor', 'treatment', 'claim'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in insurance_keywords)

    def _is_grade_specific_query(self, query: str) -> bool:
        """Check if query is specifically about a grade or requires grade context"""
        grade_keywords = [
            'my grade', 'for my grade', 'as a g', 'g1', 'g2', 'g3', 'g4', 'g5',
            'grade specific', 'my level', 'for my level', 'my benefits',
            'my allowance', 'my salary', 'my perks', 'for g1', 'for g2',
            'for g3', 'for g4', 'for g5', 'health insurance', 'medical coverage',
            'insurance policy', 'benefits package'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in grade_keywords)

    def _is_gender_specific_query(self, query: str) -> bool:
        """Check if query is specifically about gender or requires gender context"""
        gender_keywords = [
            'maternity', 'paternity', 'parental leave', 'gender', 'male', 'female',
            'woman', 'women', 'man', 'men', 'mother', 'father', 'his', 'hers',
            'expecting', 'pregnancy', 'pregnant', 'baby', 'child birth', 'son', 'daughter'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in gender_keywords)

    def _exact_word_search(self, query: str, chunks: List[str]) -> List[tuple]:
        """Perform exact word search and return chunks with scores"""
        print(f"🔍 Performing exact word search for: '{query}'")
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_words = set(chunk.lower().split())
            matches = len(query_words.intersection(chunk_words))
            if matches > 0:
                score = matches / len(query_words)
                scored_chunks.append((score, i, chunk))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        print(f"📊 Found {len(scored_chunks)} chunks with exact word matches")
        return scored_chunks

    def _hybrid_search(self, query: str, user_data: Dict) -> str:
        """Combine exact word search and semantic search for best results"""
        print(f"🔀 Starting hybrid search for: '{query}'")
        
        if not self.data_loader or not self.data_loader.all_chunks:
            return "No policy documents available."
        
        # Get all document chunks (excluding user data sheets)
        policy_chunks = []
        for doc in self.data_loader.all_chunks:
            if not doc.metadata.get('source', '').startswith('sheet:'):
                policy_chunks.append(doc.page_content)
        
        if not policy_chunks:
            return "No policy documents found."
        
        print(f"📚 Searching through {len(policy_chunks)} policy chunks")
        
        # Step 1: Exact word search
        exact_matches = self._exact_word_search(query, policy_chunks)
        
        # Step 2: Semantic search
        semantic_matches = self._semantic_search(query, policy_chunks, top_k=10)
        
        # Step 3: Combine and rank results
        combined_results = {}
        
        # Add exact matches with higher weight
        for exact_score, idx, chunk in exact_matches[:5]:
            combined_results[idx] = {
                'chunk': chunk,
                'exact_score': exact_score,
                'semantic_score': 0,
                'combined_score': exact_score * 2
            }
        
        # Add semantic matches
        for sem_score, idx, chunk in semantic_matches:
            if idx in combined_results:
                combined_results[idx]['semantic_score'] = sem_score
                combined_results[idx]['combined_score'] = (combined_results[idx]['exact_score'] * 2) + sem_score
            else:
                combined_results[idx] = {
                    'chunk': chunk,
                    'exact_score': 0,
                    'semantic_score': sem_score,
                    'combined_score': sem_score
                }
        
        # Sort by combined score
        sorted_results = sorted(combined_results.values(), 
                              key=lambda x: x['combined_score'], reverse=True)
        
        # Filter for relevance threshold
        relevant_chunks = []
        for result in sorted_results:
            if result['combined_score'] > 0.1:
                relevant_chunks.append(result['chunk'])
                if len(relevant_chunks) >= 5:
                    break
        
        print(f"✅ Found {len(relevant_chunks)} relevant chunks")
        
        if relevant_chunks:
            context = "\n\n---POLICY SECTION---\n\n".join(relevant_chunks)
            return self._generate_context_with_prompt(context, query, user_data)
        else:
            return "No relevant policy information found in the documents."

    def _generate_context_with_prompt(self, context: str, query: str, user_data: Dict) -> str:
        """Generate context with user-specific prompt instructions"""
        print("🎯 Generating context with user-specific instructions")
        
        grade = user_data.get('grade', '').strip()
        gender = user_data.get('gender', '').strip().lower()
        
        # Build prompt instructions based on query type
        instructions = []
        
        if self._is_grade_specific_query(query) and grade:
            instructions.append(f"Focus specifically on policies for grade {grade} employees")
            
        if self._is_gender_specific_query(query) and gender:
            if gender == 'female':
                instructions.append("Focus on maternity leave and women-specific policies")
            else:
                instructions.append("Focus on paternity leave and men-specific policies")
        
        if self._is_insurance_query(query) and grade:
            instructions.append(f"Provide exact coverage amounts and benefits for grade {grade}")
        
        if instructions:
            context += "\n\n---USER-SPECIFIC INSTRUCTIONS---\n\n"
            context += "\n".join(instructions)
        
        return context

    def _generate_response_prompt(self, query: str, context: str, user_data: Dict) -> str:
        """Generate a comprehensive prompt for the LLM"""
        print(f"📝 Generating response prompt for user query")
        
        username = user_data.get('username', 'Employee')
        grade = user_data.get('grade', 'not specified')
        gender = user_data.get('gender', 'not specified').lower()
        
        # Determine if we should include grade/gender info
        include_grade = self._is_grade_specific_query(query) or self._is_insurance_query(query)
        include_gender = self._is_gender_specific_query(query) and gender != 'not specified'
        
        prompt = f"""**HR Policy Assistant - GenITeam Solutions**

**Employee Profile:**
- Name: {username}
{f"- Grade: {grade}" if include_grade else ""}
{f"- Gender: {gender}" if include_gender else ""}

**Employee Query:** {query}

**Relevant Policy Context:**
{context}

**Response Instructions:"""
        
        if include_grade or include_gender:
            prompt += """
1. Provide accurate information based on the policy context"""
            if include_grade:
                prompt += f"""
2. Specifically mention the benefits and coverage for grade {grade}"""
            if include_gender:
                prompt += f"""
3. For gender-specific policies, consider the employee's gender ({gender})"""
            prompt += """
4. Include exact coverage amounts and dependent information if available
5. If information isn't available, direct to HR
6. Maintain professional tone"""
        else:
            prompt += """
1. Provide general policy information based on the context
2. Maintain a professional and helpful tone
3. If information isn't available, direct to HR"""

        prompt += """

**Professional HR Response:**"""
        
        return prompt

    def query_policy(self, query: str, username: str = None, user_data: Dict = None) -> str:
        """Main method to query the policy system"""
        print(f"\n🔍 Processing policy query: '{query[:50]}...'")
        
        if not self.llm_ready:
            return "System is initializing, please wait..."
        
        try:
            # Get user data
            if username and not user_data:
                user_data = self.get_user_data(username)
                if not user_data:
                    print(f"⚠️ User '{username}' not found in database")
                    user_data = {'username': username, 'grade': 'Unknown', 'gender': 'not specified'}
            elif not user_data:
                user_data = {'username': 'Employee', 'grade': 'Unknown', 'gender': 'not specified'}
            
            print(f"👤 User profile: {user_data.get('username', 'Unknown')} "
                  f"(Grade: {user_data.get('grade', 'N/A')}, "
                  f"Gender: {user_data.get('gender', 'not specified')})")
            
            # Get context using hybrid search
            context = self._hybrid_search(query, user_data)
            
            if "No relevant policy information" in context or "No policy documents" in context:
                return self._generate_polite_fallback(query, user_data)
            
            # Generate response
            prompt = self._generate_response_prompt(query, context, user_data)
            
            if self.llm_type == "langchain":
                response = self.llm.invoke(prompt)
            else:
                response_obj = self.llm.generate_content(prompt)
                response = response_obj.text
            
            # Post-process response
            processed_response = self._post_process_response(response, user_data, query)
            
            print("✅ Query processed successfully")
            logging.info(f"Query processed for user {user_data.get('username', 'Unknown')}: {query[:50]}...")
            
            return processed_response
            
        except Exception as e:
            logging.error(f"Query processing failed: {str(e)}")
            print(f"❌ Query processing failed: {str(e)}")
            return self._generate_polite_fallback(query, user_data)

    def _post_process_response(self, response: str, user_data: Dict, query: str) -> str:
        """Post-process the LLM response for accuracy and user-specific details"""
        print("🔧 Post-processing response...")
        
        processed = response.strip()
        grade = user_data.get('grade', '').lower()
        gender = user_data.get('gender', '').lower()
        
        # Ensure grade-specific info is accurate
        if self._is_grade_specific_query(query) and grade:
            if 'g1' in grade and 'g2' in processed.lower():
                processed = processed.replace('g2', 'g1').replace('G2', 'G1')
            elif 'g2' in grade and 'g1' in processed.lower():
                processed = processed.replace('g1', 'g2').replace('G1', 'G2')
        
        # Gender-specific corrections
        if self._is_gender_specific_query(query):
            if gender == 'female' and 'paternity' in processed.lower():
                processed = processed.replace('paternity', 'maternity').replace('Paternity', 'Maternity')
            elif gender == 'male' and 'maternity' in processed.lower():
                processed = processed.replace('maternity', 'paternity').replace('Maternity', 'Paternity')
        
        return processed

    def _generate_polite_fallback(self, query: str, user_data: Dict) -> str:
        """Generate a polite response when information is not found"""
        print("💬 Generating polite fallback response")
        
        username = user_data.get('username', 'Employee')
        
        return (f"Hello {username}, I couldn't find specific information about '{query}' in our current policy documents. "
                f"For personalized assistance, please contact HR directly at {getattr(self.config, 'HR_EMAIL', 'hr@geniteam.com')}.")

    def setup_qa_chain(self) -> None:
        """Setup the QA chain for LangChain integration"""
        print("\n⛓ Setting up QA chain...")
        if not self.llm_ready:
            raise Exception("LLM not initialized")
        
        prompt = ChatPromptTemplate.from_template(
            """Answer this HR query for {grade} employee (Gender: {gender}):
            
            Query: {query}
            
            Policy Context: {context}
            
            Provide accurate information based on their grade and gender.
            If information is not available, politely direct to HR.
            
            Response:"""
        )
        
        def get_context_wrapper(inputs):
            query = inputs["query"]
            user_data = inputs.get("user_data", {})
            return self._hybrid_search(query, user_data)
        
        if hasattr(self, 'llm_type') and self.llm_type == "langchain":
            print("🔗 Using LangChain-native GoogleGenerativeAI")
            self.chain = (
                {
                    "context": get_context_wrapper,
                    "query": RunnablePassthrough(),
                    "grade": lambda x: x.get("user_data", {}).get("grade", "N/A"),
                    "gender": lambda x: x.get("user_data", {}).get("gender", "not specified")
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            print("🔗 Using Google AI SDK with wrapper")
            def gemini_runnable(prompt_value):
                try:
                    if hasattr(prompt_value, 'to_string'):
                        prompt_text = prompt_value.to_string()
                    elif hasattr(prompt_value, 'text'):
                        prompt_text = prompt_value.text
                    elif isinstance(prompt_value, str):
                        prompt_text = prompt_value
                    else:
                        prompt_text = str(prompt_value)
                    
                    response = self.llm.generate_content(prompt_text)
                    return response.text
                    
                except Exception as e:
                    print(f"❌ Gemini generation failed: {str(e)}")
                    return f"Error generating response: {str(e)}"
            
            self.chain = (
                {
                    "context": get_context_wrapper,
                    "query": RunnablePassthrough(),
                    "grade": lambda x: x.get("user_data", {}).get("grade", "N/A"),
                    "gender": lambda x: x.get("user_data", {}).get("gender", "not specified")
                }
                | prompt
                | RunnableLambda(gemini_runnable)
                | StrOutputParser()
            )
        
        print("✅ QA chain setup completed")

    def invoke_chain(self, query: str, user_data: dict = None) -> str:
        """Invoke the QA chain with query and user data"""
        if not self.chain:
            raise Exception("QA chain not initialized. Call setup_qa_chain() first.")
        
        inputs = {
            "query": query,
            "user_data": user_data or {}
        }
        
        try:
            return self.chain.invoke(inputs)
        except Exception as e:
            print(f"❌ Chain invocation failed: {str(e)}")
            username = user_data.get('username') if user_data else None
            return self.query_policy(query, username, user_data)

    def list_users(self) -> List[str]:
        """List all users in the database"""
        return list(self.user_database.keys())

    def get_user_stats(self) -> Dict:
        """Get statistics about loaded users"""
        total_users = len(self.user_database)
        grades = {}
        genders = {}
        
        for user_data in self.user_database.values():
            grade = user_data.get('grade', 'Unknown')
            gender = user_data.get('gender', 'Unknown')
            
            grades[grade] = grades.get(grade, 0) + 1
            genders[gender] = genders.get(gender, 0) + 1
        
        return {
            'total_users': total_users,
            'grades': grades,
            'genders': genders
        }

    def debug_user_data(self, username: str) -> Dict:
        """Debug method to check user data"""
        user = self.get_user_data(username)
        print(f"\n🔍 Debugging user data for: {username}")
        print("Current user database keys:", list(self.user_database.keys()))
        print("Found user data:", user)
        
        # Check all sheets for this user
        if not user:
            print("\n🔍 Searching all sheets for user data...")
            for doc in self.data_loader.all_chunks:
                if doc.metadata.get('source', '').startswith('sheet:'):
                    try:
                        content = doc.page_content
                        if username.lower() in content.lower():
                            print(f"\nFound in sheet: {doc.metadata['source']}")
                            print("Content snippet:")
                            print(content[:500] + "..." if len(content) > 500 else content)
                    except:
                        continue
                    
        return user