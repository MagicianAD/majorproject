from django.db import models

# Create your models here.
class Contact(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField(null=True,blank=True)

    def __str__(self):
        return self.message


class Reply(models.Model):
    question = models.ForeignKey(Contact,on_delete=models.CASCADE)
    reply = models.TextField(null=True,blank=True)

    def __str__(self):
        return self.question.message