library(shiny)
library(caret)
library(ROCR)
library(plotly)
library(DT)

ui <- fluidPage(
  titlePanel("Logistic Regression App"),
  
  sidebarLayout(
    sidebarPanel(
      fileInput("datafile", "Upload CSV Data"),
      uiOutput("target_var"),
      uiOutput("exclude_columns"),
      sliderInput("train_ratio", "Training Data Ratio", min = 0.5, max = 0.9, value = 0.7),
      actionButton("run_model", "Run Model"),
      actionButton("predict_outcomes", "Predict Outcomes with New Data"),
      fileInput("newdatafile", "Upload New Data (for Prediction)", accept = ".csv"),
      downloadButton("download_predictions", "Download Predictions")
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Model Summary", verbatimTextOutput("model_summary")),
        tabPanel("Confusion Matrix", verbatimTextOutput("cm_table")),
        tabPanel("ROC Curve", plotlyOutput("roc_plot")),
        tabPanel("Probability Curve", plotlyOutput("prob_plot")),
        tabPanel("Prediction Histogram", plotlyOutput("prediction_histogram")),
        tabPanel("Actual vs Predicted", plotlyOutput("actual_vs_predicted")),
        tabPanel("Cook's Distance", plotOutput("cooks_plot")),  # Added Cook's Distance tab here
        tabPanel("Predicted Outcomes", DTOutput("pred_table"))  # Ensure this tab for predictions is included
      )
    )
  )
)



server <- function(input, output, session) {
  data <- reactive({
    req(input$datafile)
    df <- read.csv(input$datafile$datapath)
    df <- na.omit(df)  # Remove rows with NAs
    return(df)
  })
  
  model <- reactiveVal(NULL)  # Reactive value to store the model
  predictions_data <- reactiveVal(NULL)  # Reactive value to store predictions
  cooks_distance <- reactiveVal(NULL)  # Reactive value to store Cook's Distance
  
  # Dynamic UI for selecting target variable
  output$target_var <- renderUI({
    req(data())
    selectInput("target", "Select Target (Binary) Variable", choices = names(data()))
  })
  
  # Dynamic UI for excluding columns
  output$exclude_columns <- renderUI({
    req(data())
    checkboxGroupInput("excluded_cols", "Select Columns to Exclude", choices = names(data()), selected = NULL)
  })
  
  observeEvent(input$run_model, {
    req(data(), input$target)
    dataset <- data()
    target_var <- input$target
    excluded_cols <- input$excluded_cols
    
    # Exclude selected columns
    dataset <- dataset[, !names(dataset) %in% excluded_cols]
    
    # Ensure the target variable is included
    if (!target_var %in% names(dataset)) {
      showModal(modalDialog(
        title = "Error",
        "The selected target variable was excluded. Please revise your selections.",
        easyClose = TRUE
      ))
      return()
    }
    
    # Ensure the target variable is binary
    if (length(unique(dataset[[target_var]])) != 2) {
      showModal(modalDialog(
        title = "Error",
        "The selected target variable is not binary. Please select a binary variable.",
        easyClose = TRUE
      ))
      return()
    }
    
    # Split dataset into training and testing sets
    set.seed(123)
    trainIndex <- createDataPartition(dataset[[target_var]], p = input$train_ratio, list = FALSE)
    trainData <- dataset[trainIndex, ]
    testData <- dataset[-trainIndex, ]
    
    trainData <- na.omit(trainData)
    testData <- na.omit(testData)
    
    if (nrow(trainData) == 0 || nrow(testData) == 0) {
      showModal(modalDialog(
        title = "Error",
        "After removing missing values, there is not enough data to train or test the model.",
        easyClose = TRUE
      ))
      return()
    }
    
    # Logistic regression model
    fit_model <- train(as.formula(paste(target_var, "~ .")), data = trainData, method = "glm", family = "binomial",
                       trControl = trainControl(method = "cv", number = 5, selectionFunction = "best"))
    
    model(fit_model)  # Save the model in the reactive value
    
    # Cook's Distance calculation
    cooks_distance_val <- cooks.distance(fit_model$finalModel)
    cooks_distance(cooks_distance_val)  # Save Cook's Distance in the reactive value
    
    output$model_summary <- renderPrint({
      summary(fit_model$finalModel)
    })
    
    # AUC Curve
    prob <- predict(fit_model, testData, type = "prob")[, 2]
    pred <- prediction(prob, testData[[target_var]])
    perf <- performance(pred, "tpr", "fpr")
    
    output$roc_plot <- renderPlotly({
      auc <- performance(pred, measure = "auc")@y.values[[1]]
      roc_data <- data.frame(FPR = perf@x.values[[1]], TPR = perf@y.values[[1]])
      plot_ly(data = roc_data, x = ~FPR, y = ~TPR, type = 'scatter', mode = 'lines', name = "ROC Curve") %>%
        layout(title = paste("AUC Curve (AUC =", round(auc, 3), ")"), xaxis = list(title = "False Positive Rate"),
               yaxis = list(title = "True Positive Rate"))
    })
    
    output$prob_plot <- renderPlotly({
      testData$Predicted_Probability <- prob
      plot_ly(testData, x = ~Predicted_Probability, y = ~as.numeric(as.factor(testData[[target_var]])) - 1,
              type = 'scatter', mode = 'markers', color = ~testData[[target_var]]) %>%
        layout(title = "Probability Curve",
               xaxis = list(title = "Predicted Probability"),
               yaxis = list(title = "Actual Outcome (Binary)"))
    })
    
    output$prediction_histogram <- renderPlotly({
      testData$Predicted_Probability <- prob
      testData$Category <- as.factor(testData[[target_var]])
      plot_ly(testData, x = ~Predicted_Probability, color = ~Category, type = "histogram", opacity = 0.6) %>%
        layout(title = "Prediction Histogram",
               xaxis = list(title = "Predicted Probability"),
               yaxis = list(title = "Count"))
    })
    
    output$actual_vs_predicted <- renderPlotly({
      testData$Predicted_Probability <- prob
      testData$Category <- as.factor(testData[[target_var]])
      testData$Match <- ifelse((testData$Category == "Y" & prob > 0.5) | 
                                 (testData$Category == "N" & prob <= 0.5), "Correct", "Incorrect")
      testData$Match <- factor(testData$Match, levels = c("Correct", "Incorrect"))
      plot_ly(testData, x = ~Category, y = ~Predicted_Probability, color = ~Match, 
              colors = c("darkgreen", "darkred"), type = "scatter", mode = "markers", 
              marker = list(size = 10, opacity = 0.6)) %>%
        layout(
          title = "Actual vs Predicted (Colored by Match)",
          xaxis = list(title = "Actual Category"),
          yaxis = list(title = "Predicted Probability"),
          showlegend = TRUE,
          legend = list(title = list(text = "Prediction Match"))
        )
    })
    
    predicted_class <- ifelse(prob > 0.5, "Y", "N")
    cm <- confusionMatrix(factor(predicted_class), factor(testData[[target_var]]))
    output$cm_table <- renderPrint({
      cm
    })
    
    predictions_data(testData)
  })
  
  observeEvent(input$predict_outcomes, {
    req(input$newdatafile)
    fit_model <- model()
    
    if (is.null(fit_model)) {
      showModal(modalDialog(
        title = "Error",
        "Model not found. Please run the logistic regression first.",
        easyClose = TRUE
      ))
      return()
    }
    
    newdata <- read.csv(input$newdatafile$datapath)
    newdata <- na.omit(newdata)
    
    excluded_cols <- input$excluded_cols
    newdata <- newdata[, !names(newdata) %in% excluded_cols]
    
    if (nrow(newdata) == 0) {
      showModal(modalDialog(
        title = "Error",
        "The uploaded new data contains only missing values. Please upload valid data.",
        easyClose = TRUE
      ))
      return()
    }
    
    predictions <- predict(fit_model, newdata, type = "prob")
    
    output$pred_table <- renderDT({
      pred_table <- cbind(newdata, Predicted_Probability = predictions[, 2])
      head(pred_table, 10)  # Show only first 10 rows
    })
  })
  
  # Download predictions as CSV
  output$download_predictions <- downloadHandler(
    filename = function() {
      paste("predictions_", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(predictions_data(), file)
    }
  )
  
  # Render Cook's Distance Plot
  output$cooks_plot <- renderPlot({
    cooks_dist <- cooks_distance()
    plot(cooks_dist, type = "h", main = "Cook's Distance", ylab = "Cook's Distance")
    abline(h = 4 / length(cooks_dist), col = "red")  # Add threshold line
  })
}


shinyApp(ui = ui, server = server)
