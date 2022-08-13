package spacro

import jjm.DotKleisli

import io.circe._
import io.circe.syntax._
import io.circe.{Encoder, Decoder}
import io.circe.DecodingFailure
import io.circe.Json
import io.circe.JsonObject
import io.circe.generic.JsonCodec


package object sample {

  // in shared code, you should define a Prompt and Response data type for each task.
  // They should be serializable and you should not expect to have to change these often;
  // all HIT data will be written with serialized versions of the prompts and responses.
  // A good rule of thumb is to keep the minimal necessary amount of information in them.

  @JsonCodec case class Prompt(
        id: String, question: String
  )

  @JsonCodec case class Response(
        annotations: List[Json]
  )

  // you must define a task key (string) for every task, which is unique to that task.
  // this will be used as a URL parameter to access the right client code, websocket flow, etc.
  // when interfacing between the client and server.
  val validationTaskKey = "validation"

  // You then may define your API datatypes for the ajax and websocket APIs (if necessary).

  // Response sent to client from server
  @JsonCodec case class GenerationAjaxResponse(sentence: String)

  // Request sent to server from client
  @JsonCodec case class GenerationAjaxRequest(prompt: Prompt) {
    type Out = GenerationAjaxResponse
  }

  object GenerationAjaxRequest {
    import spacro.tasks._

    // These values provide a way to, given an AJAX request, encode/decode its response
    // to/from JSON. This is necessary for implementing the AJAX communication
    // between client and server.
    implicit val generationAjaxRequestDotDecoder = new DotKleisli[Decoder, GenerationAjaxRequest] {
      def apply(request: GenerationAjaxRequest) = implicitly[Decoder[request.Out]]
    }
    implicit val generationAjaxRequestDotEncoder = new DotKleisli[Encoder, GenerationAjaxRequest] {
      def apply(request: GenerationAjaxRequest) = implicitly[Encoder[request.Out]]
    }
  }
  
  // Response sent to client from server
  @JsonCodec case class ValidationAjaxResponse(sentence: String)

  // Request sent to server from client
  @JsonCodec case class ValidationAjaxRequest(prompt: Prompt) {
    type Out = ValidationAjaxResponse
  }

  object ValidationAjaxRequest {
    import spacro.tasks._

    // These values provide a way to, given an AJAX request, encode/decode its response
    // to/from JSON. This is necessary for implementing the AJAX communication
    // between client and server.
    implicit val ValidationAjaxRequestDotDecoder = new DotKleisli[Decoder, ValidationAjaxRequest] {
      def apply(request: ValidationAjaxRequest) = implicitly[Decoder[request.Out]]
    }
    implicit val ValidationAjaxRequestDotEncoder = new DotKleisli[Encoder, ValidationAjaxRequest] {
      def apply(request: ValidationAjaxRequest) = implicitly[Encoder[request.Out]]
    }
  }
  
  @JsonCodec case class QualificationAjaxResponse(sentence: String)

  // Request sent to server from client
  @JsonCodec case class QualificationAjaxRequest(prompt: Prompt) {
    type Out = QualificationAjaxResponse
  }

  object QualificationAjaxRequest {
    import spacro.tasks._

    // These values provide a way to, given an AJAX request, encode/decode its response
    // to/from JSON. This is necessary for implementing the AJAX communication
    // between client and server.
    implicit val QualificationAjaxRequestDotDecoder = new DotKleisli[Decoder, QualificationAjaxRequest] {
      def apply(request: QualificationAjaxRequest) = implicitly[Decoder[request.Out]]
    }
    implicit val QualificationAjaxRequestDotEncoder = new DotKleisli[Encoder, QualificationAjaxRequest] {
      def apply(request: QualificationAjaxRequest) = implicitly[Encoder[request.Out]]
    }
  }


}

