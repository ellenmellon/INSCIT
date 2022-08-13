package spacro.sample

import jjm.ling._
import jjm.DotKleisli

import cats.implicits._
import cats.Id

import spacro._
import spacro.tasks._
import spacro.util._
import spacro.sample._

import spacro.HITType
import spacro.tasks.NumAssignmentsHITManager
import spacro.tasks.HITManager
import spacro.tasks.Server
import spacro.tasks.TaskConfig
import spacro.tasks.TaskManager
import spacro.tasks.TaskSpecification

import akka.actor._
import akka.actor.Props
import akka.stream.scaladsl.Flow

import scala.concurrent.duration._
import scala.language.postfixOps
import scala.io.Source

import java.io._
import java.util.ArrayList;

import com.amazonaws.services.mturk.model._
import com.amazonaws.services.mturk._

import scala.util.parsing.json._
import scala.util.Random
import scala.collection.JavaConverters._

import com.typesafe.scalalogging.StrictLogging

class SampleExperiment(
  implicit config: TaskConfig,
  annotationDataService: AnnotationDataService
) {
  //implicit val ads = annotationDataService
  // Manage qualification
  
  // 1) Approval rate should be higher than 95
  val approvalRateQualificationTypeID = "000000000000000000L0"

  val approvalRateRequirement = new QualificationRequirement()
      .withQualificationTypeId(approvalRateQualificationTypeID)
      .withComparator("GreaterThanOrEqualTo")
      .withIntegerValues(if (config.isProduction) (98) else (70))
      .withRequiredToPreview(false)

  val numApprovedHITsHighQualificationTypeID = "00000000000000000040"
  val numApprovedHITsHighRequirement = new QualificationRequirement()
      .withQualificationTypeId(numApprovedHITsHighQualificationTypeID)
      .withComparator("GreaterThanOrEqualTo")
      .withIntegerValues(if (config.isProduction) (5000) else (0))
      .withRequiredToPreview(false)
  
  val numApprovedHITsQualificationTypeID = "00000000000000000040"
  val numApprovedHITsRequirement = new QualificationRequirement()
      .withQualificationTypeId(numApprovedHITsQualificationTypeID)
      .withComparator("GreaterThanOrEqualTo")
      .withIntegerValues(if (config.isProduction) (1000) else (0))
      .withRequiredToPreview(false)

  // 2) Located in US or Canada
  val localeQualificationTypeID = "00000000000000000071"
  val localeValues = new ArrayList[Locale]()
  localeValues.add(new Locale().withCountry("US"))
  localeValues.add(new Locale().withCountry("CA"))

  val localeRequirement = new QualificationRequirement()
    .withQualificationTypeId(localeQualificationTypeID)
    .withComparator(Comparator.In)
    .withLocaleValues(localeValues)
    .withRequiredToPreview(true)

  // 4) Qualified From Pilot
  val ourQualTypeLabel = "Qualified from our qualification task (Interactive Conversational QA)"
  val ourQualTypeLabelString = "Qualified from our qualification task (Interactive Conversational QA)"

  val ourQualTypeName = "Qualification made from our qualification task (Interactive Conversational QA)"

  val ourQualType = config.service
    .listQualificationTypes(
      new ListQualificationTypesRequest()
        .withQuery(ourQualTypeName)
        .withMustBeOwnedByCaller(true)
        .withMustBeRequestable(false)
        .withMaxResults(100)
    )
    .getQualificationTypes
    .asScala
    .toList
    .find(_.getName == ourQualTypeName)
    .getOrElse {
      System.out.println("Generating our pilot qualification type...")
      config.service
        .createQualificationType(
          new CreateQualificationTypeRequest()
            .withName(ourQualTypeName)
            .withKeywords("language,english,question answering")
            .withDescription(
              "Qualification was made from our pilot task."
            )
            .withQualificationTypeStatus(QualificationTypeStatus.Active)
        )
        .getQualificationType
    }
  val ourQualTypeId = ourQualType.getQualificationTypeId

  val ourQualRequirement = new QualificationRequirement()
    .withQualificationTypeId(ourQualTypeId)
    .withComparator("Exists")
    .withRequiredToPreview(false)

  
  // Defining HIT types
  
  val validationHITType = HITType(
    title = s"Evaluation of agent system outputs in interactive Conversational QA",
    description = s"""This is the task to evaluate systems that act as an agent who responds to the user request in a given information-seeking dialogue""".trim,
    reward = 0.90,
    keywords = "language,english,question answering,dialogue,Wikipedia",
    qualRequirements = Array[QualificationRequirement](
            approvalRateRequirement,
            numApprovedHITsRequirement,
            localeRequirement,
            // ourQualRequirement
    ),
    autoApprovalDelay = 2592000L,
    assignmentDuration = 3600L // 1 hr
  )
 
  // Defining data sources
  var validationDatapoints: List[Prompt] = List()
  Source.fromFile(s"data/sources_eval.jsonl").getLines.foreach(
    line => {
      var item:Map[String, String] = JSON.parseFull(line).getOrElse("none").asInstanceOf[Map[String, String]]
      var id:String = item.getOrElse("id", "none")
      var question:String = item.getOrElse("question", "none")
      validationDatapoints = Prompt(id=id, question=question) :: validationDatapoints
    })
  var validationPrompts = validationDatapoints.toVector
  
  // Defining services, managers and actors
  lazy val validationAjaxService = new DotKleisli[Id, ValidationAjaxRequest] {
    def apply(request: ValidationAjaxRequest) =
      ValidationAjaxResponse("Sample question")
  }
 
  // We can add links here to be placed at the bottom of the head tag
  // or at the bottom of the body tag (before the main JS files) in the HTML sent to Turk
  // for the task page.
  lazy val (taskPageHeadLinks, taskPageBodyLinks) = {
    import scalatags.Text.all._
    val headLinks = List(
      link(
        rel := "stylesheet",
        href := s"https://${config.serverDomain}:${config.httpsPort}/styles/bootstrap/css/bootstrap.min.css"
      ),
      link(
        rel := "stylesheet",
        href := s"https://${config.serverDomain}:${config.httpsPort}/styles/bootstrap/css/bootstrap-theme.min.css"
      ),
      link(
        rel := "stylesheet",
        href := s"https://${config.serverDomain}:${config.httpsPort}/styles/pretty-checkbox.min.css"
      ),
      link(
        rel := "stylesheet",
        href := s"https://${config.serverDomain}:${config.httpsPort}/styles/index.css"
      )
    )
    val bodyLinks = List(
      script(
        src := "https://code.jquery.com/jquery-3.1.1.js"
      ),
      script(
        src := "https://cdnjs.cloudflare.com/ajax/libs/jquery-cookie/1.4.1/jquery.cookie.min.js",
        attr("integrity") := "sha256-1A78rJEdiWTzco6qdn3igTBv9VupN3Q1ozZNTR4WE/Y=",
        attr("crossorigin") := "anonymous"
      ),
      script(
        src := "https://cdnjs.cloudflare.com/ajax/libs/tether/1.4.0/js/tether.min.js",
        attr("integrity") := "sha384-DztdAPBWPRXSA/3eYEEUWrWCy7G5KFbe8fFjk5JAIxUYHKkDx6Qin1DkWx51bBrb",
        attr("crossorigin") := "anonymous"
      ),
      script(
        src := "https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-alpha.6/js/bootstrap.min.js",
        attr("integrity") := "sha384-vBWWzlZJ8ea9aCX4pEW3rVHjgjt7zpkNpZk+02D9phzyeVkE+jo0ieGizqPLForn",
        attr("crossorigin") := "anonymous"
      )
    )
    (headLinks, bodyLinks)
  }

  // the task specification is defined on the basis of the above fields
  lazy val validationTaskSpec =
    TaskSpecification.NoWebsockets[Prompt, Response, ValidationAjaxRequest](
      validationTaskKey,
      validationHITType,
      validationAjaxService,
      validationPrompts,
      taskPageHeadElements = taskPageHeadLinks,
      taskPageBodyElements = taskPageBodyLinks
    )

  import config.actorSystem

  lazy val validationHelper = new HITManager.Helper(validationTaskSpec)

  var validationManagerPeek: MyHITManager = null
  val validationManager: ActorRef = actorSystem.actorOf( Props {
    validationManagerPeek = new MyHITManager(
      "validation",
      validationHelper,
      _ => 3,
      20,
      validationPrompts.iterator,   
      ""
    )
    validationManagerPeek
  })
 
  val validationActor = actorSystem.actorOf(Props(
    new TaskManager(validationHelper, validationManager)
  ))

  lazy val server = new Server(List(validationTaskSpec))
  
 // used to schedule data-saves
  private[this] var schedule: List[Cancellable] = Nil

  def startSaves(interval: FiniteDuration = 1 minutes): Unit = {
    if (schedule.exists(_.isCancelled) || schedule.isEmpty) {
      schedule = List(validationManager).map(
        actor =>
          config.actorSystem.scheduler
            .schedule(2 seconds, interval, actor, SaveData)(config.actorSystem.dispatcher, actor)
      )
    }
  }

  def stopSaves = schedule.foreach(_.cancel())

  def setValHITsActive(n: Int) = {
    validationManager ! SetNumHITsActive(n)
  }
  
  import TaskManager.Message._

  def start(interval: FiniteDuration = 60 seconds) = {
    server
    startSaves()
    validationActor ! Start(interval, delay = 0 seconds)
    println("All actors have started")
  }

  def stop() = {
    validationActor ! Stop
    stopSaves
  }

  def delete() = {
    validationActor ! Delete
  }

  def expire() = {
    validationActor ! Expire
  }

  def update() = {
    server
    validationActor ! Update
  }
  
  def save() = {
    validationManager ! SaveData
  }

  // Utility functions
  def giveBonusToAllValidatorsWithPrompt(promptId: String, bonus: Double) {
    validationManagerPeek.promptIdToAssignments(promptId).foreach( asn => {
      giveBonus(asn.workerId, asn.assignmentId, bonus, "human evaluation")
    })
  }

  def giveBonus(workerId: String, assignmentId: String, bonus: Double, task: String) {
    println(s"Awarding ${bonus} USD to worker ${workerId}.")
    config.service.sendBonus(
            new SendBonusRequest()
              .withWorkerId(workerId)
              .withBonusAmount(f"$bonus%.2f")
              .withAssignmentId(assignmentId)
              .withReason(
                s"Bonus awarded for ${task}."
              ))
  }

  def grantOurQualification(workerId: String) {
    if (config.service.listAllWorkersWithQualificationType(ourQualTypeId).contains(workerId)) {
      println(s"worker ${workerId} already has our validator qualification")
    } else {
      println(s"Grant our validator qualification to worker ${workerId}")
      config.service.associateQualificationWithWorker(
              new AssociateQualificationWithWorkerRequest()
                .withQualificationTypeId(ourQualTypeId)
                .withWorkerId(workerId)
                .withIntegerValue(1)
                .withSendNotification(true)
            )
    }
  }

  def removeOurQualification(workerId: String) {
    if (!config.service.listAllWorkersWithQualificationType(ourQualTypeId).contains(workerId)) {
      println(s"worker ${workerId} has not got our qualification")
    } else {
      println(s"Remove our qualification to worker ${workerId}")
      config.service.disassociateQualificationFromWorker(
              new DisassociateQualificationFromWorkerRequest()
                .withQualificationTypeId(ourQualTypeId)
                .withWorkerId(workerId)
                .withReason(s"")
            )
    }
  }

}
