# Generated by Django 5.1.1 on 2024-11-10 07:45

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0002_alter_signup_fname_alter_signup_lname_and_more'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='SignUp',
            new_name='SignUpForm',
        ),
    ]
