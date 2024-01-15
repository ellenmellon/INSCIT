var annotation = {};
var systemLooked = [];

/*! js-cookie v3.0.0-beta.3 | MIT */
!function(e,t){"object"==typeof exports&&"undefined"!=typeof module?module.exports=t():"function"==typeof define&&define.amd?define(t):(e=e||self,function(){var n=e.Cookies,r=e.Cookies=t();r.noConflict=function(){return e.Cookies=n,r}}())}(this,function(){"use strict";var e={read:function(e){return e.replace(/(%[\dA-F]{2})+/gi,decodeURIComponent)},write:function(e){return encodeURIComponent(e).replace(/%(2[346BF]|3[AC-F]|40|5[BDE]|60|7[BCD])/g,decodeURIComponent)}};function t(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)e[r]=n[r]}return e}return function n(r,o){function i(e,n,i){if("undefined"!=typeof document){"number"==typeof(i=t({},o,i)).expires&&(i.expires=new Date(Date.now()+864e5*i.expires)),i.expires&&(i.expires=i.expires.toUTCString()),n=r.write(n,e),e=encodeURIComponent(e).replace(/%(2[346B]|5E|60|7C)/g,decodeURIComponent).replace(/[()]/g,escape);var c="";for(var u in i)i[u]&&(c+="; "+u,!0!==i[u]&&(c+="="+i[u].split(";")[0]));return document.cookie=e+"="+n+c}}return Object.create({set:i,get:function(t){if("undefined"!=typeof document&&(!arguments.length||t)){for(var n=document.cookie?document.cookie.split("; "):[],o={},i=0;i<n.length;i++){var c=n[i].split("="),u=c.slice(1).join("=");'"'===u[0]&&(u=u.slice(1,-1));try{var f=e.read(c[0]);if(o[f]=r.read(u,f),t===f)break}catch(e){}}return t?o[t]:o}},remove:function(e,n){i(e,"",t({},n,{expires:-1}))},withAttributes:function(e){return n(this.converter,t({},this.attributes,e))},withConverter:function(e){return n(t({},this.converter,e),this.attributes)}},{attributes:{value:Object.freeze(o)},converter:{value:Object.freeze(r)}})}(e,{path:"/"})});

var INPUT_CONTEXT_PREFIX =              $("<div></div>").append("<h4><strong>Ongoing Dialogue:</strong></h4>");

// The following are for validation task errors:
var UNANSWERED_QUESTION =               "Error: Make sure you answer all questions.";
var NO_MOST_COMPREHENSIVE =             "Error: You need to select the most comprehensive system response(s). If you think all three responses are equally comprehensive, you should select all of them as the 'most comprehensive' and leave nothing selected as the 'least comprehensive'.";
var NO_LEAST_COMPREHENSIVE =            "Error: You need to select the least comprehensive system response(s), unless you think all three responses are equally comprehensive, in which case you should select all of them as the 'most comprehensive'.";
var BOTH_MOST_AND_LEAST_COMPREHENSIVE = "Error: No system response can be both the most and the least comprehensive one.";
var NOT_ALL_SYSTEM_LOOKED =             "Error: It seems like you answered all questions with at least one system output not even looked at. Please make sure you read all system outputs before submitting your answers.";

/*
************* Functions for loading user and agent generation tasks **************
*/


function loadInitAnnotationForm() {
  getAnnotations();
}


function loadInputContext(divId, utterances) {
  var turn = true;
  var inputContext = $(divId);
  inputContext.append(`<button type="button" class="btn-lg" style="width:500px;border:2px solid #e7e7e7" data-toggle="collapse" data-target="#input-ongoing-dialogue">Ongoing Dialogue  &nbsp; (Click to expand / collapse) </button><br><br>`);
  //inputContext.append(INPUT_CONTEXT_PREFIX);

  var input_ongoing_dialogue_div = $("<div id='input-ongoing-dialogue'></div>");
  
  for (var i = 0; i < utterances.length; ++i){
  // for (var i = 0; i < 5; ++i){
      if(turn){
        input_ongoing_dialogue_div.append("<strong><em>User:</em></strong> " + utterances[i] + "<br>");
      }
      else{
        input_ongoing_dialogue_div.append("<strong><em>Agent:</em></strong> " + utterances[i] + "<br>");
      }
      turn = !turn;
  }

  inputContext.append(input_ongoing_dialogue_div)
}


/*
************* Functions for loading validation tasks **************
*/

function loadAllValidation() {
  
  /* Load prompt inputs */
  var prompt = JSON.parse($("#prompt").attr("value"));
  var parsed_question = JSON.parse(prompt["question"]);

  var utterances = parsed_question["utterances"];

  
  var sysOutput1 = parsed_question["system 1"];
  var sysOutput2 = parsed_question["system 2"];
  var sysOutput3 = parsed_question["system 3"];

  var response1 = sysOutput1["response"];
  var evidence1 = sysOutput1["evidence"];
  var response2 = sysOutput2["response"];
  var evidence2 = sysOutput2["evidence"];
  var response3 = sysOutput3["response"];
  var evidence3 = sysOutput3["evidence"];
  
  /*
  var response1 = "response 1";
  var evidence1 = [{"title": "", "text": "content outline1"}, {"title": "title 1", "text": "text1 text1 text1 text1 text1 text1 text1 text1 text1 text1 text1 text1 text1 text1 text1 text1 text1 text1 text1 text1 text1 text1 text1 text1 text1"}];
  var response2 = "response 2";
  var evidence2 = [{"title": "", "text": "content outline2"}, {"title": "title 1", "text": "text2"}];
  var response3 = "response 3";
  var evidence3 = [{"title": "", "text": "content outline3"}, {"title": "title 1", "text": "text3"}];
  */  
  loadValidationHIT(utterances, response1, response2, response3, evidence1, evidence2, evidence3);

  /* default loading page */
  annotation["system names"] = [sysOutput1["name"], sysOutput2["name"], sysOutput3["name"]];
  systemLooked = [false, false, false];
  annotationButtonClicked(1);

  $('input[type=radio]').change(getAnnotations);
  $('input[type=checkbox]').change(getAnnotations);


  $('#feedback').keyup(getAnnotations);

  $('#uw-checkbox').prop('checked', false);
  $('#uw-checkbox').change(function(){
    if (this.checked) {
      alert(`If you are an employee of the UW, family member of a UW employee, or UW student involved in this particular research, you cannot participate in this job. Please return your HIT.`)
      $('#submit').prop('disabled', this.checked);
    }
    getAnnotations();
  })


  $('#container').append(`
    <button type="submit" disabled id="actual-submit" style="display:none""></button>`);

  // Submitting the annotation
  $("#submit").click(function () {
    getAnnotations();
    // for AWS
    $('#actual-submit').prop('disabled', false);
    $('#actual-submit').click();
  });
}

function addEvidence(evidence, systemId) {
  for (var i = 0; i < evidence.length; ++i) {
    namestring = `system-` + String(systemId) + `-evidence-` + String(i);
    $("#system-" + String(systemId)).append(`
        <br><button type="button" class="btn btn-info" style="width:300px" data-toggle="collapse" data-target="#` + namestring + `">Evidence #` + String(i) + `   &nbsp; (Click to expand / collapse) </button><br>`)
    evidenceDiv = $(`<div id="` + namestring + `" class="collapse"></div>`)
    if (evidence[i]['title'] === '') {
      evidenceDiv.append('<br><p><strong>Content Outline:</strong></p>')
      evidenceDiv.append('<p>' + evidence[i]['text'] + '</p>')
    } else {
      evidenceDiv.append('<br><p><strong>Titles:</strong></p><p>' + evidence[i]['title'] + '</p>')
      evidenceDiv.append('<p><strong>Content:</strong></p><p>' + evidence[i]['text'] + '</p>')
    }
    $("#system-" + String(systemId)).append(evidenceDiv)
  }
}

function loadValidationHIT(utterances, response1, response2, response3, evidence1, evidence2, evidence3) {
  $("#input-context").html("");
  loadInputContext("#input-context", utterances);
  $("#system-response-1").html(response1);
  $("#system-response-2").html(response2);
  $("#system-response-3").html(response3);

  
  // TODO: add evidence
  addEvidence(evidence1, 1)
  addEvidence(evidence2, 2)
  addEvidence(evidence3, 3)

  $("#btn-system-1").click(function() {
    annotationButtonClicked(1);
    getAnnotations();
  });
  $("#btn-system-2").click(function() {
    annotationButtonClicked(2);
    getAnnotations();
  });
  $("#btn-system-3").click(function() {
    annotationButtonClicked(3);
    getAnnotations();
  });
}

function annotationButtonClicked(outputId) {
  systemLooked[outputId-1] = true;
  $('.system-btn').css("border", "0px");
  $('.system-btn').css("background", "Orange");
  $("#btn-system-" + String(outputId)).css("border", "black 5px solid");
  $("#btn-system-" + String(outputId)).css("background", "Green");
  $(".system-output").hide();
  $("#system-" + String(outputId)).show();
  getAnnotations();
}


/*
************* Functions and Utils for Reading Annotations and Showing Warnings **************
*/

function isEmptyAnswer(answer) {
  return (isNaN(answer) || answer === null || answer === undefined)
}


function getAnnotations() {
  var validated = true;
  var validated_msg = "";
  var warning_msg = "";


  var isUW = $('#uw-checkbox').prop('checked');

  if (isUW) {
    $('#submit').prop('disabled', true);
    $('#validated-hint').html("If you are an employee of the UW, family member of a UW employee, or UW student involved in this particular research, you cannot participate in this job. Please return your HIT.");
    return;
  }

  /* Reading all annotation values */
  var utility1 = parseInt($('input[name=utility-1]:checked').val());
  var utility2 = parseInt($('input[name=utility-2]:checked').val());
  var utility3 = parseInt($('input[name=utility-3]:checked').val());
  var consistency1 = parseInt($('input[name=consistency-1]:checked').val());
  var consistency2 = parseInt($('input[name=consistency-2]:checked').val());
  var consistency3 = parseInt($('input[name=consistency-3]:checked').val());
  var coherence1 = parseInt($('input[name=coherence-1]:checked').val());
  var coherence2 = parseInt($('input[name=coherence-2]:checked').val());
  var coherence3 = parseInt($('input[name=coherence-3]:checked').val());
  var fluency1 = parseInt($('input[name=fluency-1]:checked').val());
  var fluency2 = parseInt($('input[name=fluency-2]:checked').val());
  var fluency3 = parseInt($('input[name=fluency-3]:checked').val());


  var most_comprehensive_1 = $('#compare-most-1').prop('checked')
  var most_comprehensive_2 = $('#compare-most-2').prop('checked')
  var most_comprehensive_3 = $('#compare-most-3').prop('checked')
  var least_comprehensive_1 = $('#compare-least-1').prop('checked')
  var least_comprehensive_2 = $('#compare-least-2').prop('checked')
  var least_comprehensive_3 = $('#compare-least-3').prop('checked')

  var allRadioButtonAnswers = [utility1, utility2, utility3, consistency1, consistency2, consistency3, coherence1, coherence2, coherence3, fluency1, fluency2, fluency3];
  if (allRadioButtonAnswers.map(x => isEmptyAnswer(x)).indexOf(true)>-1) {
    validated = false;
    validated_msg = UNANSWERED_QUESTION;
  }

  
  if (validated) {
    if (!most_comprehensive_1 && !most_comprehensive_2 && !most_comprehensive_3) {
      validated = false;
      validated_msg = NO_MOST_COMPREHENSIVE;
    } else if ((!most_comprehensive_1 || !most_comprehensive_2 || !most_comprehensive_3) && (!least_comprehensive_1 && !least_comprehensive_2 && !least_comprehensive_3)) {
      validated = false;
      validated_msg = NO_LEAST_COMPREHENSIVE;
    } else if ((most_comprehensive_1 && least_comprehensive_1) || 
      (most_comprehensive_2 && least_comprehensive_2) ||
      (most_comprehensive_3 && least_comprehensive_3)) {
      
      validated = false;
      validated_msg = BOTH_MOST_AND_LEAST_COMPREHENSIVE;
    }
  }

  if (validated && systemLooked.indexOf(false)>-1) {
    validated = false;
    validated_msg = NOT_ALL_SYSTEM_LOOKED;
  }
  

  if ($.trim($('#feedback').val())==="") {
    $('#feedback').hide();
  } else {
    $('#feedback').show();
  }

  if(validated) {
    warning_msg += "<br/><em><big>You are allowed to submit now!</big></em> "
    if ($.trim($('#feedback').val())==="") {
      warning_msg += "<strong><em><big>If you have any feedback for us, please leave in the feedback box.</big></em></strong>"
    }
    $('#feedback').show();
  }

  $('#submit').prop('disabled', !validated);
  if (warning_msg === "") {
    $('#validated-hint').html(validated_msg);
  } else if (validated_msg === "") {
    warning_msg = "<p style='color:Blue'>" + warning_msg + "</p>";
    $('#validated-hint').html(warning_msg);
  } else {
    warning_msg = "<p style='color:Blue'>" + warning_msg + "</p>"
    $('#validated-hint').html(validated_msg + '<br/>' + warning_msg);
  }

  if (validated) {
    annotation["utility1"] = utility1;
    annotation["utility2"] = utility2;
    annotation["utility3"] = utility3;
    annotation["consistency1"] = consistency1;
    annotation["consistency2"] = consistency2;
    annotation["consistency3"] = consistency3;
    annotation["coherence1"] = coherence1;
    annotation["coherence2"] = coherence2;
    annotation["coherence3"] = coherence3;
    annotation["fluency1"] = fluency1;
    annotation["fluency2"] = fluency2;
    annotation["fluency3"] = fluency3;
    annotation["most_comprehensive_1"] = most_comprehensive_1;
    annotation["most_comprehensive_2"] = most_comprehensive_2;
    annotation["most_comprehensive_3"] = most_comprehensive_3;
    annotation["least_comprehensive_1"] = least_comprehensive_1;
    annotation["least_comprehensive_2"] = least_comprehensive_2;
    annotation["least_comprehensive_3"] = least_comprehensive_3;
  
    $('#response').val(JSON.stringify({'annotations': [annotation]}));
  }
}

