
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD

object LogisticRegression extends App {
  object RegType extends Enumeration {
    type RegType = Value
    val NONE, L1, L2 = Value
  }

  object DataFormat extends Enumeration {
    type DataFormat = Value
    val EntryList, LibSVM = Value
  }

  import RegType._
  import DataFormat._

  case class Params(
      input: String = null,
      numIterations: Int = 100,
      stepSize: Double = 1.0,
      regType: RegType = L2,
      regParam: Double = 0.01,
      dataFormat: DataFormat = EntryList) extends AbstractParams[Params]

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
    sc.stop()
  }

}
