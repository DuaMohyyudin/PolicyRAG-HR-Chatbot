from typing import Dict, Optional, Any
from datetime import datetime

class QueryHandler:
    def __init__(self, authenticated_user: Dict, rag_system, authenticator, config):
        self.user = authenticated_user
        self.rag_system = rag_system
        self.authenticator = authenticator
        self.config = config
        self.hr_contact = config.HR_EMAIL
        self.leave_types = {
            'casual': 'Casual Leave (CL)',
            'sick': 'Sick Leave (SL)',
            'annual': 'Annual Leave (AL)',
            'maternity': 'Maternity Leave',
            'paternity': 'Paternity Leave'
        }

    def _refresh_user_data(self) -> bool:
        """Refresh user data with proper error handling"""
        try:
            updated_user = self.authenticator.get_authenticated_user()
            if not updated_user:
                raise ValueError("User data refresh returned empty")
            self.user = updated_user
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to refresh user data: {str(e)}")

    def _detect_leave_type(self, query: str) -> Optional[str]:
        """Detect specific leave type from query text"""
        query = query.lower()
        for leave_type, display_name in self.leave_types.items():
            if leave_type in query or display_name.lower() in query:
                return display_name
        
        # Special cases
        if "expecting" in query or "pregnancy" in query:
            user_gender = self.user.get('gender', '').lower()
            return "Maternity Leave" if user_gender == 'female' else "Paternity Leave"
        return None

    def _normalize_leave_type(self, raw_type: str) -> Optional[str]:
        """Normalize leave type input to standard types with validation"""
        if not raw_type:
            return None
            
        raw_type = raw_type.lower()
        for standard_type, display_name in self.leave_types.items():
            if standard_type in raw_type or display_name.lower() in raw_type:
                return display_name
        return None

    def _validate_leave_application(self, leave_type: str, days: float) -> Optional[str]:
        """Validate leave application parameters"""
        user_gender = self.user.get('gender', '').lower()
        
        # Type validation
        if not leave_type:
            return "Leave type must be specified"
            
        # Gender-specific validation
        if "maternity" in leave_type.lower() and user_gender != 'female':
            return "Maternity leave is only available for female employees"
        if "paternity" in leave_type.lower() and user_gender != 'male':
            return "Paternity leave is only available for male employees"
            
        # Duration validation
        if days <= 0:
            return "Leave days must be positive"
        if days < self.config.MIN_LEAVE_DAYS:
            return f"Minimum leave is {self.config.MIN_LEAVE_DAYS} day"
        if days > self.config.MAX_LEAVE_DAYS:
            return f"Maximum leave is {self.config.MAX_LEAVE_DAYS} days"
            
        return None

    def _handle_leave_application(self, query: str) -> str:
        """Process leave application with comprehensive validation"""
        try:
            parts = query.split()
            if len(parts) < 4:
                return "Please specify leave type and duration (e.g., 'apply for sick leave 2')"
                
            leave_type = ' '.join(parts[3:-1]) if len(parts) > 4 else parts[3]
            days = float(parts[-1]) if parts[-1].replace('.', '').isdigit() else 1.0
            
            # Normalize and validate
            normalized_type = self._normalize_leave_type(leave_type)
            if not normalized_type:
                return f"Invalid leave type. Available types: {', '.join(self.leave_types.values())}"
                
            validation_error = self._validate_leave_application(normalized_type, days)
            if validation_error:
                return validation_error
                
            # Apply for leave
            success = self.authenticator.apply_for_leave(
                username=self.user['username'],
                days=days,
                leave_type=normalized_type
            )
            
            if success:
                self._refresh_user_data()
                return (
                    f"✅ {normalized_type} application for {days} day(s) submitted.\n"
                    f"Remaining leaves: {self.user.get('remaining_leaves', 'N/A')}\n"
                    f"Status: Pending approval"
                )
            return "Failed to submit leave application"
                
        except ValueError as e:
            return f"Invalid input: {str(e)}"
        except Exception as e:
            return f"System error: {str(e)}"

    def _handle_leave_query(self, query: str) -> str:
        """Handle leave-related queries with context"""
        # Check for balance inquiries
        if "balance" in query or "remaining" in query:
            return f"Your current leave balance: {self.user.get('remaining_leaves', 'N/A')} days"
            
        # Check for specific leave type
        leave_type = self._detect_leave_type(query)
        if leave_type:
            # Get policy info with user context
            response = self.rag_system.query_policy(
                f"{leave_type} policy",
                user_data=self.user
            )
            return f"{response}\nYour current balance: {self.user.get('remaining_leaves', 'N/A')} days"
            
        # General leave policy query
        return self.rag_system.query_policy(query, user_data=self.user)

    def _handle_policy_query(self, query: str) -> str:
        """Handle general policy queries with RAG"""
        return self.rag_system.query_policy(query, user_data=self.user)

    def _handle_benefits_query(self, query: str) -> str:
        """Handle benefits-related queries with grade context"""
        grade = self.user.get('grade', '')
        enhanced_query = f"{query} for grade {grade}" if grade else query
        response = self.rag_system.query_policy(enhanced_query, user_data=self.user)
        
        if "not found" in response.lower():
            return f"{response}\nFor detailed benefits, contact {self.hr_contact}"
        return response

    def _handle_general_query(self, query: str) -> str:
        """Handle all other queries with fallback logic"""
        # Try policy lookup first
        policy_response = self.rag_system.query_policy(query, user_data=self.user)
        if "not found" not in policy_response.lower():
            return policy_response
            
        # Classify query type
        if any(term in query.lower() for term in ["contact", "email", "phone"]):
            return f"Please contact HR at {self.hr_contact}"
        elif any(term in query.lower() for term in ["salary", "pay", "compensation"]):
            return "For salary queries, please contact payroll@company.com"
            
        return (
            "I couldn't find relevant information in our policies. "
            f"Please contact {self.hr_contact} for assistance."
        )

    def handle_query(self, query: str) -> str:
        """Main query processing with error handling"""
        try:
            query = query.lower().strip()
            self._refresh_user_data()
            
            if query.startswith("apply for leave"):
                return self._handle_leave_application(query)
                
            # Route queries
            if any(term in query for term in ["leave", "time off", "vacation"]):
                return self._handle_leave_query(query)
            elif any(term in query for term in ["policy", "rule", "guideline"]):
                return self._handle_policy_query(query)
            elif any(term in query for term in ["benefit", "perk", "allowance"]):
                return self._handle_benefits_query(query)
            else:
                return self._handle_general_query(query)
                
        except Exception as e:
            return f"System error occurred. Please contact {self.hr_contact} if the issue persists."