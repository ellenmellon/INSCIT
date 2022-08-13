package spacro.sample

import cats.implicits._

import spacro._
import spacro.tasks._

import scala.collection.mutable
import scala.util.{Failure, Success, Try}

import akka.actor.ActorRef

import com.amazonaws.services.mturk.model
import com.amazonaws.services.mturk.model.AssociateQualificationWithWorkerRequest
import com.amazonaws.services.mturk.model.AssignmentStatus
import com.amazonaws.services.mturk.model.DisassociateQualificationFromWorkerRequest
import com.amazonaws.services.mturk.model.ListAssignmentsForHITRequest

import com.typesafe.scalalogging.StrictLogging

import io.circe.{Encoder, Decoder}
import io.circe.syntax._
import scala.collection.JavaConverters._

class MyHITManager(
  keyword: String,
  helper: HITManager.Helper[Prompt, Response],
  numAssignmentsForPrompt: Prompt => Int,
  initNumHITsToKeepActive: Int,
  _promptSource: Iterator[Prompt],
  doneQualificationTypeId: String
)(
  implicit annotationDataService: AnnotationDataService,
  config: TaskConfig
  //settings: QASRLSettings
) extends NumAssignmentsHITManager[Prompt, Response](
    helper,
    numAssignmentsForPrompt,
    initNumHITsToKeepActive,
    _promptSource,
    true
  ) with StrictLogging {

  import helper._
  import config._
  import taskSpec.hitTypeId

  val promptIdToAssignmentsFilename:String = keyword + "PromptIdToAssignments"
  var promptIdToAssignments = {
    annotationDataService
      .loadLiveData(promptIdToAssignmentsFilename)
      .map(_.mkString)
      .map(x => io.circe.parser.decode[Map[String, List[Assignment[Response]]]](x).right.get)
      .toOption
      .getOrElse {
        Map.empty[String, List[Assignment[Response]]]
      }
  }

  override def reviewAssignment(
    hit: HIT[Prompt],
    assignment: Assignment[Response]
  ): Unit = {
    evaluateAssignment(hit, startReviewing(assignment), Approval(""))
  
    val prevAssignments = promptIdToAssignments.getOrElse(hit.prompt.id, List())
    val updatedAssignments = assignment :: prevAssignments
    promptIdToAssignments = promptIdToAssignments.updated(hit.prompt.id, updatedAssignments)
 
    if (doneQualificationTypeId.length() > 0) {
      config.service.associateQualificationWithWorker(
        new AssociateQualificationWithWorkerRequest()
          .withQualificationTypeId(doneQualificationTypeId)
          .withWorkerId(assignment.workerId)
          .withIntegerValue(1)
          .withSendNotification(false))
    }
  }
  
  override lazy val receiveAux2: PartialFunction[Any, Unit] = {
    case SaveData => save
    case Pring    => println(keyword + " manager pringed.")
  }
  
  def save = {
    annotationDataService.saveLiveData(promptIdToAssignmentsFilename, promptIdToAssignments.asJson.noSpaces)
    logger.info(keyword + " data saved.")
  }

 
}
