from django.conf.urls import url
from django.contrib import admin
from django.conf import settings
from . import views
from django.conf.urls.static import static
#from django.contrib.staticfiles.urls import staticfiles_urlpatterns
admin.site.site_header= "AIA weather company"
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^contact/$', views.contact,name='contact'),
    url(r'^delete/comment/(?P<id>[0-9]+)/$', views.comment_delete,name='comment_delete'),
    url(r'^delete/reply/<int:id>/$', views.reply_delete,name='reply_delete'),
    url(r'^$',views.homepage,name='home'),
]+static(settings.STATIC_URL,document_root=settings.STATIC_ROOT)
#staticfiles_urlpatterns()
