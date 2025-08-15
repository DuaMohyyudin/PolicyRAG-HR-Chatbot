from typing import Optional, Dict
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread.exceptions import APIError
from datetime import datetime
from google.oauth2.service_account import Credentials


class Authenticator:
    def __init__(self, config):
        self.config = config
        self.user_data = None
        self.authenticated_user = None
        self.sheet = None
        self.worksheet = None

    def _connect_to_google_sheets(self):
        """Establish connection to Google Sheets"""
        try:
            scope = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            creds = Credentials.from_service_account_file(
                self.config.CREDENTIALS_PATH,
                scopes=scope
            )
            client = gspread.authorize(creds)
            self.sheet = client.open_by_key(self.config.GOOGLE_SHEETS_ID)
            self.worksheet = self.sheet.worksheet(self.config.SHEET_NAME)
            return True
        except Exception as e:
            raise Exception(f"Google Sheets connection failed: {str(e)}")

    def load_user_data(self) -> None:
        """Load employee data from Google Sheets"""
        try:
            if not self._connect_to_google_sheets():
                raise Exception("Could not connect to Google Sheets")

            data = self.worksheet.get_all_records()
            
            if not data:
                raise ValueError("No data found in Google Sheet")
            
            self.user_data = data
        except APIError as e:
            raise Exception(f"Google Sheets API error: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to load user data: {str(e)}")

    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user credentials with real-time Google Sheets check"""
        try:
            # Refresh data before authentication
            self.load_user_data()
            
            password = int(password)
            for user in self.user_data:
                if (user['username'].lower() == username.lower() and 
                    user['password'] == password):
                    self.authenticated_user = user
                    return True
            return False
        except ValueError:
            return False
        except Exception as e:
            raise Exception(f"Authentication error: {str(e)}")

    def get_authenticated_user(self) -> Optional[Dict]:
        """Get authenticated user data with real-time refresh"""
        try:
            if self.authenticated_user:
                # Force a complete refresh of all data
                self.load_user_data()
                username = self.authenticated_user['username']
                # Find the user in the freshly loaded data
                for user in self.user_data:
                    if user['username'].lower() == username.lower():
                        self.authenticated_user = user  # Update the cached user
                        return user
        except Exception as e:
            print(f"Error refreshing user data: {str(e)}")
        return None
        """Get authenticated user data with real-time refresh"""
        try:
            if self.authenticated_user:
                # Refresh data to get latest values
                self.load_user_data()
                username = self.authenticated_user['username']
                for user in self.user_data:
                    if user['username'].lower() == username.lower():
                        return user
        except Exception:
            pass
        return None         

    def apply_for_leave(self, username: str, days: float) -> bool:
        """Apply for leave and update Google Sheet"""
        try:
            self.load_user_data()
            
            # Find the user's row
            cell = self.worksheet.find(username.lower(), in_column=1)
            row = cell.row
            
            # Get current values
            remaining_leaves = float(self.worksheet.cell(row, 4).value)
            
            # Validate leave application
            if days <= 0:
                raise ValueError("Leave days must be positive")
            if days < self.config.MIN_LEAVE_DAYS:
                raise ValueError(f"Minimum leave is {self.config.MIN_LEAVE_DAYS} day")
            if days > self.config.MAX_LEAVE_DAYS:
                raise ValueError(f"Maximum leave is {self.config.MAX_LEAVE_DAYS} days")
            if days > remaining_leaves:
                raise ValueError("Not enough remaining leaves")
            
            # Update the sheet
            new_remaining = remaining_leaves - days
            self.worksheet.update_cell(row, 4, new_remaining)
            
            # Add to leave history
            leave_history = self.sheet.worksheet("LeaveHistory") if "LeaveHistory" in [ws.title for ws in self.sheet.worksheets()] else self.sheet.add_worksheet(title="LeaveHistory", rows=100, cols=4)
            leave_history.append_row([
                username,
                str(days),
                datetime.now().strftime("%Y-%m-%d"),
                "Pending Approval"
            ])
            
            return True
        except Exception as e:
            raise Exception(f"Failed to apply for leave: {str(e)}")