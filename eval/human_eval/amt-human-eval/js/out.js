(function() {
  var ARROW = "<span class='glyphicon glyphicon-arrow-right'></span>";

  $( window ).init(function(){
    loadValidationHTML();
    setWidth();
    loadInstructions();
    loadAllValidation();
    loadInitAnnotationForm();
    turkSetAssignmentID();

  });

  function loadInstructions() {
    INSTRUCTIONS = getValidationInstructions();  // TODO

    /* For Instructions */
    $('#instruction-header')
      .html('Instructions (Click to expand).')
      .mouseover(function(){
        this.style.textDecoration = "underline";
      })
      .mouseout(function(){
        this.style.textDecoration = "none";
      })
      .click(function(){
        if ($('#instruction-body').css('display')=='block') {
          $('#instruction-body').css('display', 'none');
          $('#instruction-header').html('Instructions (Click to expand).');
        } else {
          $('#instruction-body').css('display', 'block');
          $('#instruction-header').html('Instructions (Click to collapse).');
        }
      });
    $('#instruction-body').css('display', 'none');

    $('.instructions-item').click(function() {
      $('.active').removeClass("active");
      $('#'+this.id).parent().addClass("active");
      $('#instructions').html(INSTRUCTIONS[this.id]);
    });
    $('#instructions').html(INSTRUCTIONS['instructions-overview']);
  }

  function getSystemHtml(systemId) {
    return `<div class="system-output" id="system-` + String(systemId) + `" style="display:none">
        <br>
        <p style="color:Blue"><strong><big>Response:</big></strong></p>
        <p id="system-response-` + String(systemId) + `"> </p><hr>
        <p style="color:Blue"><strong><big>Evidence Paragraphs:</big></strong></p>
    </div>`;
  }

  function getUtilityEvalButtons(systemId) {
    var namestring = `utility-` + String(systemId);
    return `<br/><label style="color:DarkBlue"> - System ` + String(systemId) + `:  &nbsp;  </label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-2" name="` + namestring + `" value="2"  /> <span> All </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-1" name="` + namestring + `" value="1"  /> <span> Most </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-0" name="` + namestring + `" value="0"  /> <span> Half </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-neg-1" name="` + namestring + `" value="-1"  /> <span> Very Few </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-neg-2" name="` + namestring + `" value="-2"  /> <span> None </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-no" name="` + namestring + `" value="-10"  /> <span> No Evidence Predicted </span></label>`
  }

  function getConsistencyEvalButtons(systemId) {
    var namestring = `consistency-` + String(systemId);
    return `<br/><label style="color:DarkBlue"> - System ` + String(systemId) + `: &nbsp; </label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-2" name="` + namestring + `" value="2"  /> <span> All </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-1" name="` + namestring + `" value="1"  /> <span> Most </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-0" name="` + namestring + `" value="0"  /> <span> Some </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-neg-1" name="` + namestring + `" value="-1"  /> <span> Very Little </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-neg-2" name="` + namestring + `" value="-2"  /> <span> None </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-no" name="` + namestring + `" value="-10"  /> <span> No Factual Info In the Response </span></label>`
  }

  function getCoherenceEvalButtons(systemId) {
    var namestring = `coherence-` + String(systemId);
    return `<br/><label style="color:DarkBlue"> - System ` + String(systemId) + `: &nbsp; </label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-2" name="` + namestring + `" value="2"  /> <span> Very </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-1" name="` + namestring + `" value="1"  /> <span> Somewhat </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-neg-1" name="` + namestring + `" value="-1"  /> <span> Somewhat not </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-neg-2" name="` + namestring + `" value="-2"  /> <span> Very not </span></label>`
  }

  function getFluencyEvalButtons(systemId) {
    var namestring = `fluency-` + String(systemId);
    return `<br/><label style="color:DarkBlue"> - System ` + String(systemId) + `: &nbsp; </label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-2" name="` + namestring + `" value="2"  /> <span> Very </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-1" name="` + namestring + `" value="1"  /> <span> Somewhat </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-neg-1" name="` + namestring + `" value="-1"  /> <span> Somewhat not </span></label>
    <label style="font-weight:normal"><input type="radio" id="` + namestring + `-neg-2" name="` + namestring + `" value="-2"  /> <span> Very not </span></label>`
  }

  function getEvalHtml() {
    return `<div>
            <!-------------------------------
               Evidence Utility
            --------------------------------->

            <div class="form-group" id="utility">
              <label> <big>How many evidence paragraphs are <em style="color:Chocolate">properly used</em> in each system response?</big></label>
              ` + getUtilityEvalButtons(1) + getUtilityEvalButtons(2) + getUtilityEvalButtons(3) + `
            </div>


            <!-------------------------------
               Response Factual Consistency
            --------------------------------->
            
            <div class="form-group" id="consistency">
              <label> <big>How much factual information in the system response is <em style="color:Chocolate">grounded</em> in AND <em style="color:Chocolate">consistent</em> with its evidence paragraphs or the dialogue context, ignoring commonsense knowledge?</big></label>
              ` + getConsistencyEvalButtons(1) + getConsistencyEvalButtons(2) + getConsistencyEvalButtons(3) + `
            </div>

            <!-------------------------------
               Response Coherence
            --------------------------------->

            <div class="form-group" id="coherence">
              <label> <big>How <em style="color:Chocolate">coherent</em> is the response with the given dialogue context?</big></label>
              ` + getCoherenceEvalButtons(1) + getCoherenceEvalButtons(2) + getCoherenceEvalButtons(3) + `
            </div>

            <!-------------------------------
               Response Fluency
            --------------------------------->
   
            <div class="form-group" id="fluency">
              <label> <big>How <em style="color:Chocolate">fluent</em> is the response?</big></label>
              ` + getFluencyEvalButtons(1) + getFluencyEvalButtons(2) + getFluencyEvalButtons(3) + `
            </div>

            <!-------------------------------
               Response Comprehensiveness
            --------------------------------->

            <div class="form-group" id="compare-most-form">
              <label> <big>Which system response is the <em style="color:Chocolate">most comprehensive</em> one?</big></label><br>
              <input type="checkbox" class="compare-most" id="compare-most-1"></input> System 1 &nbsp; &nbsp;
              <input type="checkbox" class="compare-most" id="compare-most-2"></input> System 2 &nbsp; &nbsp;
              <input type="checkbox" class="compare-most" id="compare-most-3"></input> System 3
            </div>

            <div class="form-group" id="compare-least-form">
              <label> <big>Which system response is the <em style="color:Chocolate">least comprehensive</em> one?</big></label><br>
              <input type="checkbox" class="compare-least" id="compare-least-1"></input> System 1 &nbsp; &nbsp;
              <input type="checkbox" class="compare-least" id="compare-least-2"></input> System 2 &nbsp; &nbsp;
              <input type="checkbox" class="compare-least" id="compare-least-3"></input> System 3
            </div>
            
            </div>`
  }

  function loadValidationHTML() {
    $('#taskContent').html(
      `<div class="container" id="container" role="main">

        <!-- Instruction -->
        <div class="panel panel-default">
          <div class="panel-heading"><button id="instruction-header" type="button" class="btn-lg" ></button></div>
          <div class="panel-body" id="instruction-body">
            <nav class="navbar navbar-default">
              <div class="container-fluid">
                <ul class="nav navbar-nav">
                  <li class="active"><a href="#" id="instructions-overview" class="instructions-item">Overview</a></li>
                  <li><a href="#" id="instructions-step-by-step" class="instructions-item">FAQ</a></li>
                  <li><a href="#" id="instructions-bonuses" class="instructions-item">Qualification</a></li>
                </ul>
              </div>
            </nav>
            <div id="instructions">
              Instructions (TODO)
            </div>
          </div>
        </div>

        <!-- Main Content in 2 Columns -->
        <div class="row">

          <!-- Left column -->
          <div class="col col-12 col-md-7"> 
            <!-- User Input for Step 1 -->
            <div class="panel panel-default narrow-panel">
              <div class="panel-body" id="input-context">(Loading...)</div>
            </div>
            
            <!-- User Input for Step 2 -->`
            + getEvalHtml() +
            `
            <!-- Optional User Feedback -->
            <br />
            <p id="validated-hint" class="small-hint red" style="color:Red;"></p>
            <p id="pay-hint" class="small-hint"></p>
            <textarea placeholder="Feedback (Optional)" class="" rows="4" id="feedback" name="feedback"></textarea>
            <br /><br />
            <input type="checkbox" id="uw-checkbox"></input>  <span class="small">Are you an employee of the UW, family member of a UW employee, or UW student involved in this particular research?</span><br />
            <div id="submit-button-div">
              <button type="button" class="btn btn-primary" id="submit">Submit!</button>
            </div>
          </div>

          <!-- Right column -->
          <div class="col col-12 col-md-5">
            <button type="button" class="btn btn-primary system-btn" id="btn-system-1">System Output 1</button>
            <button type="button" class="btn btn-primary system-btn" id="btn-system-2">System Output 2</button>
            <button type="button" class="btn btn-primary system-btn" id="btn-system-3">System Output 3</button>
            `
              + getSystemHtml(1) + getSystemHtml(2) + getSystemHtml(3) +
            `   
          </div>

        </div>
      </div>`);
  }

  function getValidationInstructions() {
    return {"instructions-overview": 
    `<big><strong>Please read the instructions thoroughly before beginning. Full understanding of the instructions will help you to retain your qualification (see Qualification section).</strong></big><br /><br /><hr />
    <p>
      In each task, you will be given an ongoing user-agent conversation with the last turn from the user, as well as three systems' prediction outputs for the next agent turn (including the agent response and textual evidence paragraphs from Wikipedia).
      <strong>Your goal is to fully understand the user’s current request and rate / evaluate every dimension of each system output as follows: </strong>
    </p>
    <br />
    
    <p>
      <strong style="color:Blue;"><em>Evidence Paragraph Utility - </em></strong>
      An evidence paragraph is considered as <strong>properly used</strong> by the corresponding agent response, if it contains <strong>at least one piece of (non-repetitive) factual information that is used in the response without any mis-interpretation</strong>. Evaluate whether all, most, half, few or none of evidence paragraphs predicted by each system are properly used. Any redundant or fully mis-interpreted evidence paragraph should NOT be considered as properly used (see FAQ). Select "No Evidence Predicted" if no evidence is predicted from that system. 
    </p>

    <p>
      <strong style="color:Blue;"><em>Response Factual Consistency - </em></strong>
      Decide how much factual information in each system-predicted response <strong>can find evidence in the annotated paragraphs (plus their titles) or the dialogue context, without any mis-interpretation</strong>. You can ignore minor commonsense knowledge when doing the count. For example, "a bicyle has two wheels" or "1+1=2" is clearly commonsense. If you think some knowledge might be commonly known in a specific region (e.g., US), but not necessarily known in other parts of the world, you should not count that as commonsense (e.g., Seattle is a rainy city). We understand this can be subjective, and simply ask you to follow your best judgment. Select "No Factual Info In the Response" if the response is something like <span class="q">Sorry, I didn't find the answer for you.</span>
    </p>

    <p>
      <strong style="color:Blue;"><em>Response Coherence - </em></strong>
      Decide whether each system-predicted response is coherent to the dialogue context and appropriate for the current user request. For example, a response may be considered <strong>very coherent</strong> if it tries to address the user request (e.g., provides direct answer, relevant information; raises a clarification; informs the user of no answer found) without any request mis-interpretation. Minor chit-chat is ok. Minor but reasonable mis-interpretation of the user request can be thought as <strong>somewhat coherent</strong>. Major mis-interpretation or off-topic responses can be considered as <strong>somewhat or very incoherent</strong>. The response coherence should also be decided based on the dialogue flow. For example, if the previous agent offers multiple options to provide an answer but the annotated agent response indicates no answer nor relevant information even after the user selects an option, it may appear incoherent. Too many consecutive agent turns with a clarification question may also seem incoherent. <strong>Note that this dimension should be rated independently of response factuality.</strong> 
    </p>

    <p>
      <strong style="color:Blue;"><em>Response Fluency - </em></strong>
      Decide whether each system-predicted response is fluent * independent * of the dialogue context. Focus on the language only (e.g., grammar). 
    </p>

    <p>
      <strong style="color:Blue;"><em>Response Comprehensiveness Comparison - </em></strong>
      Decide which system response contains <strong>the most and least comprehensive answer scope</strong>.
      <strong>Note that this is * NOT * to simply compare which response contains more information.</strong> Please carefully read the following notes: 
      <ul>
      <li>Side information for a specific answer should be ignored in this comparison. For example, given there are multiple answers (answer A and B) to the user request, if response 1 is a direct answer containing many details about answer A while response 2 is a direct answer containing both A and B without any side information or asks a clarification "Do you want to know about answer A or B?", <strong>response 2</strong> should be considered as more comprehensive as it covers a broader scope. </li>
      <li>You should only focus on the most relevant AND factually consistent information. Factually inconsistent / ungrounded information should NOT be counted. HOWEVER, they should also not contribute to negative effect to the evaluation. Just simply ignore them. That being said, responses like "I don't find the answer" should always be the least comprehensive one as it contains zero information.</li>
      <li>A response containing relevant information to the user request only should normally be considered as less comprehensive than a response with a direct answer or a clarification. <strong>However</strong>, sometimes it can be tricky to tell whether the information is an answer or just relevant information. You should just follow your best judgment. Relevant information should always be considered as more comprehensive than a response with no information (e.g., <span class="q">Sorry, I didn't find the answer for you.</span>). </li>
      </ul>
    </p>
  </p>

  <hr />

  <p>
  <h4><strong><span class="warn">IMPORTANT NOTES !!!</span></strong></h4>
  1. Sometimes, you may find an evidence paragraph to be the <strong>content outline</strong> of a Wikipedia page, which simply means the system finds the page structure layout (and titles) to be enough for providing evidence without going to the content details. Treat it as a normal evidence paragraph and do the evaluations.
  <br>  <br style="line-height:8px;"/>
  2. <strong>If you find multiple responses to have the same comprehensiveness</strong> (or incomparable as they approach the user request from different angles), reflect that by selecting multiple systems as the most / least comprehensive ones. If you think all three systems responses are equally comprehensivess, then mark all of them as the most comprehensive ones and leave the least comprehensive one as empty.
  <br style="line-height:8px;"/>
  
  
  <hr><hr>
  <span style="color:Blue">
  See the <strong>FAQ</strong> and <strong>Qualification</strong> tabs for more explanations of common mistakes and our bonus structure. <br/><br/>
  </span>
  <span><em>If you run into any issue, please email me at <a href="mailto: zeqiuwu1@uw.edu"><strong>zeqiuwu1@uw.edu</strong></a> or DM me (<strong>Ellen Wu</strong>) on Slack. I'll respond within a few minutes (at most 1 hour) during 8am-10pm PT (Sun-Fri).</em></span>`,  
  
  'instructions-step-by-step': `
    <ol>
    <li>
      How can I tell whether an evidence paragraph is redundant or not?
      ` + ARROW + ` Each annotated paragraph (or title) should provide evidence to at least one unique piece of information in the response, which cannot be found in another annotated evidence paragraph. Otherwise, it would be counted as redundant.
      Let's say your response contains information A and B. You find paragraph 1 containing A, paragraph 2 containing both A and B and paragraph 3 containing C only. In this case, you should ONLY label paragraph 2 as the single evidence paragraph.
    </li>
    <br/>
    <li>
      Can you give examples of minor or major mis-interpretations of evidence?
      ` + ARROW + ` For example, providing <span class="q">"over 200 feet deep"</span> as the answer of the user question 
      <span class="q">"How deep is the Great Barrier Reef?"</span> is a major mis-interpretation of the evidence sentence <span class="q">"The reef is located in the Coral Sea, off the coast of Queensland, Australia, separated from the coast 
      by a channel 100 miles wide in places and over 200 feet deep."</span>. <br> Providing <span class="q">"XXX"</span> as the answer of the user question 
      <span class="q">"How did the Great Depression end?"</span> is a minor mis-interpretation of the evidence sentence <span class="q">"The common view among economic historians is that XXX leads to the end of the Great Depression."</span> 
      For the same question, if the agent response uses the exact same sentence "The common view ... ", the dialogue situation can be annotated as either "Short answer found and can be directly provided" or "No direct answer but relevant info / partial answer found".
    </li>
    </ol>`,
    "instructions-bonuses": `For each HIT, we have multiple assignments to different workers and will take the majority scores as the final evaluation. <strong>We will constantly monitor whether your evaluation scores mostly align with the majority ones. If we notice that your evaluation frequently diverges from other workers', we will have to recall your qualification.</strong>
    <br><br>
    <strong>Aligning with majority scores does not mean they have to exactly match.</strong> For example, we may consider selecting "All" and "Most" for the second question as being aligned due to the question's subjectivity.`};
  }

  function setWidth() {
    $('#container').width($('#taskContent').width()-100);
  
    $(".panel").width($('#container').width());
    $(".input-group").width($('#container').width());
    $(".row").width($('#container').width());

    $(".narrow-panel").width($('#container').width()*1/2+50);
    $(".narrow-input-group").width($('#container').width()*1/2);

    $("#feedback").width($('#container').width()/3);
    
  }

})();


