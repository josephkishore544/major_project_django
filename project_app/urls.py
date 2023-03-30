from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import *

urlpatterns = [
    path("", homepage, name='homepage'),
    path("generate/", generate_home, name='generate_home'),
    path("generate/result/", generate_result, name = 'generate_result'),
    path("manipulate/", manipulate_home, name = 'manipulate_home'),
    path("manipulate/upload/", manipulate_upload, name = 'manipulate_upload'),
    path("manipulate/last/", manipulate_last, name = 'manipulate_last'),
    path("manipulate/result/", manipulate_result, name = 'manipulate_result'),
]

if settings.DEBUG :
    urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)