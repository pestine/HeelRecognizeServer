from django.http import HttpResponse

def test(request):
    res = request
    print(request)
    return HttpResponse('Receive request:{}'.format(res))
