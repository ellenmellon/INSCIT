import spacro._
import spacro.tasks._
import spacro.util._
import spacro.sample._
import com.amazonaws.services.mturk.model._
import com.amazonaws.services.mturk._

import java.io._
import java.nio.file.{Files, Path, Paths}

val interface = "0.0.0.0"
val domain = "qa.cs.washington.edu"   // TODO: change
val projectName = "INSCIT_Eval"
val httpsPort = 7781   // TODO: change
val httpPort = 7782    // TODO: change
val isProduction = false   // TODO: change to true for actual eval
val currentName = "main"

val annotationPath = Paths.get(s"data/${currentName}/annotations")
implicit val config: TaskConfig = {
  if(isProduction) {
    val hitDataService = new FileSystemHITDataService(annotationPath.resolve("production"))
    ProductionTaskConfig(projectName, domain, interface, httpPort, httpsPort, hitDataService)
  } else {
    val hitDataService = new FileSystemHITDataService(annotationPath.resolve("sandbox"))
    SandboxTaskConfig(projectName, domain, interface, httpPort, httpsPort, hitDataService)
  }
}

val liveDataPath = Paths.get(s"data/example/${currentName}/live")
implicit val annotationDataService = new FileSystemAnnotationDataService(liveDataPath)
val exp = new SampleExperiment

def yesterday = {
  val cal = java.util.Calendar.getInstance
  cal.add(java.util.Calendar.DATE, -1)
  cal.getTime
}

import scala.collection.JavaConverters._

def expireHITById(hitId: String) = {
  config.service.updateExpirationForHIT(
    (new UpdateExpirationForHITRequest)
      .withHITId(hitId)
      .withExpireAt(yesterday))
}

def displayAllHITs() = {
  config.service.listAllHITs.foreach( hit => {
    println(hit.getQuestion())
  })
}

def approveAllAssignmentsByHITId(hitId: String) = for {
  mTurkAssignment <- config.service.listAssignmentsForHIT(
    new ListAssignmentsForHITRequest()
      .withHITId(hitId)
      .withAssignmentStatuses(AssignmentStatus.Submitted)
    ).getAssignments.asScala.toList
} yield config.service.approveAssignment(
  new ApproveAssignmentRequest()
    .withAssignmentId(mTurkAssignment.getAssignmentId)
    .withRequesterFeedback(""))

def deleteHITById(hitId: String) =
  config.service.deleteHIT((new DeleteHITRequest).withHITId(hitId))

def disableHITById(hitId: String) = {
  expireHITById(hitId)
  deleteHITById(hitId)
}

def getActiveHITIds = {
  config.service.listAllHITs.map(_.getHITId)
}

def approveAllAssignments = {
  getActiveHITIds.foreach( hitId => { approveAllAssignmentsByHITId(hitId) } )
}

def reset = {
  getActiveHITIds.foreach( d => {
    approveAllAssignmentsByHITId(d)
    expireHITById(d)
    deleteHITById(d)
  })
}

def resetExceptUnassignable = {
  config.service.listAllHITs.foreach(hit => {
    if (hit.getHITStatus()!="Unassignable") {
      approveAllAssignmentsByHITId(hit.getHITId)
      expireHITById(hit.getHITId)
      deleteHITById(hit.getHITId)
    }
  })
}

def getHITStatuses: Map[String, List[(String, Int)]] = {
  var statuses: List[(String, String, Int)] = List()
  config.service.listAllHITs.foreach(hit => {
    val hitId = hit.getHITId()
    val status = hit.getHITStatus()
    val numSubmitted = config.service.listAssignmentsForHIT(new ListAssignmentsForHITRequest()
      .withHITId(hitId)
      .withAssignmentStatuses(AssignmentStatus.Submitted))
    .getAssignments.asScala.toList.size
    statuses = (hit.getHITTypeId, status, numSubmitted) :: statuses
  })
  statuses.groupBy(x => x._1).map(_ match {
    case (k, v) => k -> v.map(x => (x._2, x._3))
  })
}

def getApprovalStatuses: Map[String, List[(String, Int)]] = {
  var statuses: List[(String, String, Int)] = List()
  config.service.listAllHITs.foreach(hit => {
    val hitId = hit.getHITId()
    val status = hit.getHITStatus()
    val numSubmitted = config.service.listAssignmentsForHIT(new ListAssignmentsForHITRequest()
      .withHITId(hitId)
      .withAssignmentStatuses(AssignmentStatus.Approved))
    .getAssignments.asScala.toList.size
    statuses = (hit.getHITTypeId, status, numSubmitted) :: statuses
  })
  statuses.groupBy(x => x._1).map(_ match {
    case (k, v) => k -> v.map(x => (x._2, x._3))
  })
}

def messageWorkers(subject: String, text: String, workerIds: List[String]) = {
  config.service.notifyWorkers(
    new NotifyWorkersRequest()
      .withSubject(subject)
      .withMessageText(text)
      .withWorkerIds(workerIds.asJava))
  println(s"Send Message titled '${subject}' to ${workerIds.size} workers")
}

import cats.implicits._
import scala.io.Source
import scala.util.parsing.json._
import scala.math

var workerFilenames = List("workerIds/workers.json")

def getWorkers(filenames: List[String]) = {
  var allWorkers: List[String] = List()
  filenames.foreach (filename => {
    println(s"Loading ${filename}...")
    Source.fromFile(filename).getLines.foreach( line => {
      val workers = JSON.parseFull(line).getOrElse(List()).asInstanceOf[List[String]]
      workers.foreach(worker => {
        allWorkers = worker :: allWorkers
      })
    })
  })
  allWorkers
}

def grantQualification(workerId: String) = {
  exp.grantOurQualification(workerId)
}

def getWorkers(filenames: List[String]) = {
  var allWorkers: List[List[String]] = List()
  filenames.foreach (filename => {
    println(s"Loading ${filename}...")
    Source.fromFile(filename).getLines.foreach( line => {
      val workers = JSON.parseFull(line).getOrElse(List()).asInstanceOf[List[String]]
      for( index <- 0 until math.ceil(workers.size.toDouble/50).toInt ){
        val start = index*50
        val end = math.min((index+1)*50, workers.size)
        allWorkers = workers.slice(start, end) :: allWorkers
      }
    })
  })
  allWorkers
}

def exit = {
  // actor system has to be terminated for JVM to be able to terminate properly upon :q
  config.actorSystem.terminate
  // flush & release logging resources
  import org.slf4j.LoggerFactory
  import ch.qos.logback.classic.LoggerContext
  LoggerFactory.getILoggerFactory.asInstanceOf[LoggerContext].stop
  System.out.println("Terminated actor system and logging. Type :q to end.")
}

exp.server
