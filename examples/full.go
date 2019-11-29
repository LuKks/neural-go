package main

import (
  "fmt"
  "time"
  "github.com/lukks/neural-go"
)

const DebugColor = "\033[0;36m%s\033[0m"

func main () {
  rgb := neural.NewNeural([]neural.Layer{
    neural.NewLayer(32, 3),
    neural.NewLayer(1, 0),
  })

  rgb.InputRange(0, 0.0, 255.0)
  rgb.InputRange(1, 0.0, 255.0)
  rgb.InputRange(2, 0.0, 255.0)
  rgb.OutputRange(0, 0.0, 1.0)

  fmt.Printf(DebugColor, "benchmark\n")
  benchmark(&rgb)

  fmt.Printf(DebugColor, "think some values\n")
  fmt.Printf("255, 255, 255 [1.0] -> %f\n", rgb.Think([]float64{ 255.0, 255.0, 255.0 }))
  fmt.Printf("0  , 0  , 0   [0.0] -> %f\n", rgb.Think([]float64{ 0.0, 0.0, 0.0 }))

  fmt.Printf(DebugColor, "learning\n")
  for i := 0; i <= 5000; i++ {
    var rate float64 = 0.1
    var mse float64

    light := []float64{ 1.0 }
    dark := []float64{ 0.0 }

    mse += rgb.LearnRaw([]float64{ 1.0, 0.0, 0.0 }, []float64{ 1.0 }, rate)
    mse += rgb.Learn([]float64{ 0.0, 255.0, 0.0 }, light, rate)
    mse += rgb.Learn([]float64{ 0.0, 0.0, 255.0 }, light, rate)
    mse += rgb.Learn([]float64{ 0.0, 0.0, 0.0 }, dark, rate)
    mse += rgb.Learn([]float64{ 100.0, 100.0, 100.0 }, light, rate)
    mse += rgb.Learn([]float64{ 107.0, 181.0, 255.0 }, dark, rate)
    mse += rgb.Learn([]float64{ 0.0, 53.0, 105.0 }, dark, rate)
    mse += rgb.Learn([]float64{ 150.0, 150.0, 75.0 }, light, rate)
    mse += rgb.Learn([]float64{ 75.0, 75.0, 0.0 }, dark, rate)
    mse += rgb.Learn([]float64{ 0.0, 75.0, 75.0 }, dark, rate)
    mse += rgb.Learn([]float64{ 150.0, 74.0, 142.0 }, light, rate)
    mse += rgb.Learn([]float64{ 50.0, 50.0, 75.0 }, dark, rate)
    mse += rgb.Learn([]float64{ 103.0, 22.0, 94.0 }, dark, rate)
    mse /= 13;
    
    if mse < 0.01 {
      fmt.Printf("mse threshold on iter %v\n", i)
      break
    }

    if i % 1000 == 0 {
      fmt.Printf("iter %v, mse %f\n", i, mse)
    }
  }

  fmt.Printf(DebugColor, "think some values\n")
  fmt.Printf("255, 255, 255 [1.0] -> %f\n", rgb.Think([]float64{ 255.0, 255.0, 255.0 }))
  fmt.Printf("0  , 0  , 0   [0.0] -> %f\n", rgb.Think([]float64{ 0.0, 0.0, 0.0 }))

  fmt.Printf(DebugColor, "think new values\n")
  fmt.Printf("243, 179, 10  [1.0] -> %f\n", rgb.ThinkRaw([]float64{ 0.952, 0.701, 0.039 }))
  fmt.Printf("75 , 50 , 50  [0.0] -> %f\n", rgb.Think([]float64{ 75.0, 50.0, 50.0 }))
  fmt.Printf("95 , 99 , 104 [1.0] -> %f\n", rgb.Think([]float64{ 95.0, 99.0, 104.0 }))
  fmt.Printf("65 , 38 , 70  [0.0] -> %f\n", rgb.Think([]float64{ 65.0, 38.0, 70.0 }))
}

func benchmark (rgb *neural.Neural) {
  var start, end int64

  start = nowMillis()
  for i := 0; i < 200000; i++ { // this takes 0.164 seconds
    // then this takes ~0.00000082s
    rgb.ThinkRaw([]float64{ 1.0, 1.0, 1.0 })
  }
  end = nowMillis()
  fmt.Printf("think raw %v millis\n", end - start)

  start = nowMillis()
  for i := 0; i < 200000; i++ { // this takes 0.184 seconds
    // then this takes ~0.00000092s
    rgb.Think([]float64{ 255.0, 255.0, 255.0 })
  }
  end = nowMillis()
  fmt.Printf("think %v millis\n", end - start)
}

func nowMillis() int64 {
  return time.Now().UnixNano() / 1e6
}
