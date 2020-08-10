from . import models
from django import forms


class ImageUploadForm(forms.ModelForm):
    image = forms.ImageField(widget=forms.ClearableFileInput(attrs={'multiple': True}))

    class Meta():
        model = models.ImagesUploadModel
        fields = ('image',)