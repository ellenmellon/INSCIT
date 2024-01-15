package spacro.sample

import spacro.tasks.TaskDispatcher

/** Main class for the client; dispatches to the sample task. */
object Dispatcher extends TaskDispatcher {

  override val taskMapping = Map[String, () => Unit](
    sampleTaskKey -> (() => Client.main())
  )
}
