"""
bootstrap_admin.py - Run this ONCE, locally, to promote your own account
to admin. After that, manage roles from the Manage Users page instead.

Usage:
    py bootstrap_admin.py your-email@example.com
"""

import sys
from db import init_db, set_role


def main():
    if len(sys.argv) != 2:
        print("Usage: py bootstrap_admin.py your-email@example.com")
        sys.exit(1)

    email = sys.argv[1]
    init_db()
    set_role(email, "admin")
    print(f"{email} is now an admin. Refresh the app (or log back in) to see the change.")


if __name__ == "__main__":
    main()