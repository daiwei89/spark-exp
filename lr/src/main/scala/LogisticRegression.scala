
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
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV}
import org.apache.spark.mllib.linalg.SparseVector

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
      .action((x, c) => c.copy(regParam = x))
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
    println(s"Experiment params: $params")
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

    val trainErrorTimer = new Timer
    val prediction = model.predict(training.map(_.features))
    val predictionAndLabel = prediction.zip(training.map(_.label))
    val trainError = predictionAndLabel.map { case (p, l) =>
      if (p == l) 0
      else 1
      //math.abs(p - l)
    }.reduce(_ + _) / numTrain.toDouble
    val trainErrorTime = trainErrorTimer.elapsed
    println(s"Train error: $trainError (eval time: $trainErrorTime)")

    val trainObjTimer = new Timer
    val w_array = model.weights.toArray
    val w_brz = new BDV[Double](model.weights.toArray)
    val bias = model.intercept
    val regObj = params.regType match {
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
        0.5 * params.regParam * l2  // 1/2 * lambda * ||w||^2
    }

    val localWeights = w_brz
    val bcWeights = training.context.broadcast(localWeights)
    val logisticLoss = training.mapPartitions { iter =>
      val bias_local = bias
      val w_brz_local = bcWeights.value
      iter.map { labeledPoint =>
        val feature = labeledPoint.features.asInstanceOf[SparseVector]
        val feature_brz = new BSV[Double](feature.indices, feature.values,
          feature.size)
        val dotProd = w_brz_local.dot(feature_brz) + bias_local
        labeledPoint.label match {
          case 0 => math.log(1 + math.exp(dotProd))
          case 1 => math.log(1 + math.exp(-dotProd))
        }
      }
    }.reduce(_+_) / numTrain.toDouble
    val objValue = logisticLoss + regObj

    //val normalizedLogisticLoss = logisticLoss / numTrain.toDouble
    //val normalizedObjValue = normalizedLogisticLoss + regObj

    val trainObjTime = trainObjTimer.elapsed
    println(s"Logistic Loss: $logisticLoss; Train obj: "
      + s"$objValue; Eval time: $trainObjTime")

    sc.stop()
  }

}
