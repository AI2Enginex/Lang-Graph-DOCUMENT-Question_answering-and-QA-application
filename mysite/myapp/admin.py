from django.contrib import admin
from .models import SignUpForm
from .models import Book
# Register your models here.



# Register the SignUp model
admin.site.register(SignUpForm)

class BookAdmin(admin.ModelAdmin):
    # Display title, author, publication date, and price in the list view
    list_display = ('title', 'author', 'publication_date', 'price')

    # Add filters for the genre and publication date
    list_filter = ('genre', 'publication_date')

    # Enable searching by title and author
    search_fields = ('title', 'author')

    # Order the list by publication date, newest first
    ordering = ('-publication_date',)

    # Make the 'price' field editable directly in the list view
    list_editable = ('price',)

    # # Make the 'publication_date' field read-only in the detail view
    # readonly_fields = ('publication_date',)

# Register the Book model with the customized admin
admin.site.register(Book, BookAdmin)
