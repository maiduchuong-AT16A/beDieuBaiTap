from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def submit_test(request):
    if request.method == 'POST':
          # print('Post: "%s"' % request.POST)
          # print('Body: "%s"' % request.body)
        print(request.POST['MaNhanVien'])
    return HttpResponse("")