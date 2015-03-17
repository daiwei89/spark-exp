/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//package org.apache.spark.examples.mllib;

import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import org.apache.spark.api.java.*;
import scala.Tuple2;

/**
 * Logistic regression based classification using ML Lib.
 */
public final class LRClassifier {

  public static void main(String[] args) {
    if (args.length != 4) {
      System.err.println("Usage: LRClassifier <input_dir> <step_size> <niters> <mini_batch_fraction>");
      System.exit(1);
    }
    SparkConf sparkConf = new SparkConf().setAppName("LRClassifier");
    //scala.collection.Seq<scala.Tuple2<String,String>> env = sparkConf.getExecutorEnv();
    //JavaSparkContext sc = new JavaSparkContext(sparkConf);
    SparkContext sc = new SparkContext(sparkConf);
    long startTime = System.currentTimeMillis();
    JavaRDD<LabeledPoint> points = MLUtils.loadLibSVMFile(sc, args[0]).toJavaRDD();
    long endTime   = System.currentTimeMillis();
    long totalTime = endTime - startTime;
    System.out.println("======= Total data loading time: " + totalTime);
    //JavaRDD<String> lines = sc.textFile(args[0]);
    //JavaRDD<LabeledPoint> points = lines.map(new ParsePoint()).cache();
    double stepSize = Double.parseDouble(args[1]);
    int iterations = Integer.parseInt(args[2]);
    double miniBatchFraction = Double.parseDouble(args[3]);

    long startTimeRun = System.currentTimeMillis();
    // Another way to configure LogisticRegression
    //
    // LogisticRegressionWithSGD lr = new LogisticRegressionWithSGD();
    // lr.optimizer().setNumIterations(iterations)
    //               .setStepSize(stepSize)
    //               .setMiniBatchFraction(1.0);
    // lr.setIntercept(true);
    // LogisticRegressionModel model = lr.train(points.rdd());

    final LogisticRegressionModel model = LogisticRegressionWithSGD.train(points.rdd(),
      iterations, stepSize, miniBatchFraction);

    JavaRDD<Tuple2<Double, Double>> scoreAndLabels = points.map(
        new Function<LabeledPoint, Tuple2<Double, Double>>() {
          public Tuple2<Double, Double> call(LabeledPoint p) {
            Double score = model.predict(p.features());
            return new Tuple2<Double, Double>(score, p.label());
          }
        }
        );
    double train_error = JavaDoubleRDD.fromRDD(scoreAndLabels.map(
          new Function<Tuple2<Double, Double>, Object>() {
            public Object call(Tuple2<Double, Double> pair) {
              Double err = pair._1() - pair._2();
              if (err < 0) {
                err *= -1.;
              }
              return err;
            }
          }
          ).rdd()).mean();

    //System.out.print("Final w: " + model.weights());
    System.out.println("===== Train error: " + train_error + " numIteration: "
        + iterations);

    sc.stop();
    long endTimeRun   = System.currentTimeMillis();
    long totalTimeRun = endTimeRun - startTimeRun;
    System.out.println("======= Total algorithm run time: " + totalTimeRun);
  }
}
