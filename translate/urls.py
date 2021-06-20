from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

#import views

obj_classify_view = views.main_classification()
obj_translate_view = views.main_translate()


urlpatterns = [
    path('home/', obj_translate_view.Home),
    path('translate_comments', obj_translate_view.translate_comments, name = 'translate_comments'),
    path('filter_results', obj_translate_view.filter_results, name = 'filter_results'),
    path('trans/download', obj_translate_view.download, name = 'download'),
    path('graphs/', obj_translate_view.graphs_contact_type,name='graphs'),
    path('display_wordcloud', obj_translate_view.display_wordcloud, name = 'display_wordcloud'),
    path('all_channels', obj_translate_view.graphs_contact_type, name = 'all_channels'),
    path('channel_wise', obj_translate_view.graphs_contact_type, name = 'channel_wise')
    
    
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)