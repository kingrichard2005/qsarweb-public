﻿// Description:  This script is meant to contain client-side security logic
// see: https://docs.djangoproject.com/en/dev/ref/contrib/csrf/
function getCookie(name) {
   var cookieValue = null;
   if (document.cookie && document.cookie != '') {
      var cookies = document.cookie.split(';');
      for (var i = 0; i < cookies.length; i++) {
         var cookie = jQuery.trim(cookies[i]);
         // Does this cookie string begin with the name we want?
         if (cookie.substring(0, name.length + 1) == (name + '=')) {
            cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
            break;
         }
      }
   }
   return cookieValue;
}

function csrfSafeMethod(method) {
   // these HTTP methods do not require CSRF protection
   return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
}

$(function () {
   $.ajaxSetup({
      beforeSend: function (xhr, settings) {
         var csrftoken = getCookie('csrftoken');
         if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
            xhr.setRequestHeader("X-CSRFToken", csrftoken);
         }
      }
   });
});