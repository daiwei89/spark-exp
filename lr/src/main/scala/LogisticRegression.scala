
import org.apache.log4j.{Level, Logger}
import scopt.OptionParser
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.SparkContext._
//import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.{SimpleUpdater, SquaredL2Updater, L1Updater}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD

object LogisticRegression extends App {
  object RegType extends Enumeration {
    type RegType = Value
    val NONE, L1, L2 = Value
  }

  object DataFormat extends Enumeration {
    type DataFormat = Value
    val LibSVM = Value
  }

  import RegType._
  import DataFormat._

  case class Params(
      input: String = null,
      numIterations: Int = 100,
      stepSize: Double = 1.0,
      regType: RegType = L2,
      regParam: Double = 0.01,
      minibatchFraction: Double = 1,
      dataFormat: DataFormat = LibSVM) extends AbstractParams[Params]

  val defaultParams = Params()

  val parser = new OptionParser[Params]("LinearRegression") {
    head("LinearRegression: an example app for linear regression.")
    opt[Int]("numIterations")
      .text("number of iterations")
      .action((x, c) => c.copy(numIterations = x))
    opt[Double]("stepSize")
      .text(s"initial step size, default: ${defaultParams.stepSize}")
      .action((x, c) => c.copy(stepSize = x))
    opt[String]("regType")
      .text(s"regularization type (${RegType.values.mkString(",")}), " +
      s"default: ${defaultParams.regType}")
      .action((x, c) => c.copy(regType = RegType.withName(x)))
    opt[String]("dataFormat")
      .text(s"data format (${DataFormat.values.mkString(",")}), " +
      s"default: ${defaultParams.dataFormat}")
      .action((x, c) => c.copy(dataFormat = DataFormat.withName(x)))
    opt[Double]("regParam")
      .text(s"regularization parameter, default: ${defaultParams.regParam}")
    opt[Double]("minibatchFraction")
      .text(s"fraction of points to use per epoch: ${defaultParams.minibatchFraction}")
    arg[String]("<input>")
      .required()
      .text("input paths to data in (i) EntryList format, without .X or .Y "
        + "(2) LIBSVM format")
      .action((x, c) => c.copy(input = x))
    note(
      """
        |For example, the following command runs this app on a synthetic dataset:
        |
        |  bin/spark-submit --class org.apache.spark.examples.mllib.LinearRegression \
        |  examples/target/scala-*/spark-examples-*.jar \
        |  data/mllib/sample_linear_regression_data.txt
      """.stripMargin)
  }

  parser.parse(args, defaultParams).map { params =>
    run(params)
  } getOrElse {
    sys.exit(1)
  }

  def run(params: Params) {
    val conf = new SparkConf().setAppName(s"LinearRegression with $params")
    val sc = new SparkContext(conf)

    Logger.getRootLogger.setLevel(Level.WARN)

    val dataLoadingTimer = new Timer
    val training = params.dataFormat match {
      case LibSVM => MLUtils.loadLibSVMFile(sc, params.input).cache()
    }
    val numTrain = training.count()
    val dataLoadingTime = dataLoadingTimer.elapsed
    println(s"Data loading time: $dataLoadingTime")

    println(s"numTrain: $numTrain")
    val updater = params.regType match {
      case NONE => new SimpleUpdater()
      case L1 => new L1Updater()
      case L2 => new SquaredL2Updater()
    }

    val trainingTimer = new Timer
    val algorithm = new LogisticRegressionWithSGD()
    algorithm.optimizer
      .setNumIterations(params.numIterations)
      .setStepSize(params.stepSize)
      .setUpdater(updater)
      .setRegParam(params.regParam)
      .setMiniBatchFraction(params.minibatchFraction)
    val model = algorithm.run(training)
    val trainingTime = trainingTimer.elapsed
    val numIterations = params.numIterations
    println(s"Training time: $trainingTime ($numIterations Iterations)")

    val prediction = model.predict(training.map(_.features))
    val predictionAndLabel = prediction.zip(training.map(_.label))
    val w_array = model.weights.toArray
    var regObj = params.regType match {
      case NONE => 0
      case L1 =>
        var l1 = 0.0
        for (i <- 0 to (w_array.length - 1)) {
          l1 += math.abs(w_array(i))
        }
        params.regParam * l1
      case L2 =>
        var l2 = 0.0
        for (i <- 0 to (w_array.length - 1)) {
          l2 += w_array(i) * w_array(i)
        }
        params.regParam * l2
    }

    val logisticLoss = model.logisticLoss(training)
     .reduce(_+_) / numTrain.toDouble
    val objValue = logisticLoss + params.regParam * regObj

    val trainError = predictionAndLabel.map { case (p, l) =>
      math.abs(p - l)
    }.reduce(_ + _) / numTrain.toDouble

    println(s"Objective value = $objValue; Train Error: $trainError")

    sc.stop()
  }

}
