from django.shortcuts import render, redirect, get_list_or_404
from django.http import HttpResponse
from . import forms, models
from django.conf import settings
import os
import shutil
from . import process

# Create your views here.
def index(request):
    # if not request.session['result']:
    #     request.session['result'] = False
    #     print(request.session['result'])
    if request.method == "POST":
        request.session['result'] = False
        form = forms.ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            for i in request.FILES.getlist('image'):
                models.ImagesUploadModel.objects.create(image=i)
            return redirect('dog_breed:return_breeds')
    else:
        if not request.session['result']:
            request.session['result'] = False
            print(request.session['result'])
        form = forms.ImageUploadForm()
    return render(request, 'index.html', {'form': form})


def delete_images():
    models.ImagesUploadModel.objects.all().delete()
    folder = str(settings.MEDIA_ROOT + '\img')
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try: 
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason %s' % (file_path, e))
    # return redirect('dog_breed:index')

def predict(img_paths):
    custom_data = process.create_data_batches(img_paths)
    path = str('J:\jaimin (E)\Programming Practice\Machine Learning and Data Science\Web Projects\Dog_Breed_Prediction_Website\\full-image-set-mobilenetv2-Adam.h5')
    print(path)
    model_path = path                # "20200720-11101595243432-full-image-set-mobilenetv2-Adam.h5"
    loaded_model = process.load_model(model_path)
    custom_preds = loaded_model.predict(custom_data)
    custom_preds_labels = [process.get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
    custom_images = process.unbatchify(custom_data)
    process.plot_images(custom_images, custom_preds_labels, custom_preds)


def return_breeds(request):
    images = get_list_or_404(models.ImagesUploadModel)
    img_paths = []
    for img in images:
        img_paths.append(img.image.path.replace('\\', '/'))
    predict(img_paths)
    delete_images()
    request.session['result'] = True
    return redirect('dog_breed:index')








