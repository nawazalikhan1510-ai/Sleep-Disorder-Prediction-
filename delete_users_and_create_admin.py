#!/usr/bin/env python
"""Delete all users and create admin account"""
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'sleep_app.settings')
django.setup()

from django.contrib.auth import get_user_model

User = get_user_model()

# Step 1: Delete all users
print("=" * 60)
print("STEP 1: Deleting all existing users...")
print("=" * 60)
user_count = User.objects.count()
print(f"Found {user_count} users in database")

if user_count > 0:
    User.objects.all().delete()
    print(f"✓ Successfully deleted {user_count} users")
else:
    print("✓ No users to delete")

# Step 2: Create admin user
print("\n" + "=" * 60)
print("STEP 2: Creating new admin account...")
print("=" * 60)

try:
    admin_user = User.objects.create_superuser(
        username='admin',
        email='admin@sleepsense.com',
        password='admin123'
    )
    print("✓ Admin user created successfully!")
    
    print("\n" + "=" * 60)
    print("ADMIN CREDENTIALS")
    print("=" * 60)
    print("Username: admin")
    print("Email:    admin@sleepsense.com")
    print("Password: admin123")
    print("=" * 60)
    
    print("\nYou can now:")
    print("  • Login at: http://localhost:8000/accounts/login/")
    print("  • Use email: admin@sleepsense.com")
    print("  • Use password: admin123")
    print("  • Access admin panel: http://localhost:8000/admin/")
    print("\n")
    
except Exception as e:
    print(f"✗ Error creating admin: {e}")
    import traceback
    traceback.print_exc()

