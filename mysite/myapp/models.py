from django.db import models
from django.contrib.auth.hashers import make_password

class SignUpForm(models.Model):
    username = models.CharField(max_length=100, unique=True)
    fname = models.CharField(max_length=100)
    lname = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)  # Store hashed password

    def save(self, *args, **kwargs):
        if not self.id and self.password:  # Hash password before saving a new user
            self.password = make_password(self.password)
        super().save(*args, **kwargs)

    def check_password(self, raw_password):
        """Check if the given raw password matches the stored hashed password"""
        from django.contrib.auth.hashers import check_password
        return check_password(raw_password, self.password)
    
class Book(models.Model):
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=100)
    publication_date = models.DateField()
    genre = models.CharField(max_length=50)
    price = models.DecimalField(max_digits=6, decimal_places=2)

    def __str__(self):
        return self.title