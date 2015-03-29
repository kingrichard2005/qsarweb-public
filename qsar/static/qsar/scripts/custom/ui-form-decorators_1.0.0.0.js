// Description: Library contains methods and functions for decorating forms 
// and any other corresponding form fields.
function getMachineLearningModels() {
    $.ajax({
        url: "getMachineLearningModels/",
        type: "post",
        data: { "type": JSON.stringify($("#select-modeltype").val()) },
        cache: 'false',
        dataType: "json",
        async: 'true',
        success: function (response) {
            // debug
            // console.log(response.data)
            $('#select-machineLearningModel').html("<option value='' disabled selected>Machine Learning Models...</option>");
            // populate machine learning drop-down selector
            $.each(response.data, function (key, value) {
                $('#select-machineLearningModel')
                    .append($("<option></option>")
                    .attr("value", value)
                    .text(value));
            });

            // restore any previously saved session information
            if (response['Model_Name'] !== '') {
                //restore the model name from a previously configured session
                $("#select-machineLearningModel").val(response['Model_Name'])
            }

            $("#select-machineLearningModel").trigger("chosen:updated");
        },
        error: function (response) {
            alert("error getting results from server api getMachineLearningModels(), please report to administrators.")
        },
    });
}

function getModelTypes()
{
    // Calls backend API for model types
    $.ajax({
        url: "getModelTypes/",
        type: "post",
        dataType: "json",
        async: 'true',
        success: function (response) {
            // debug
            // console.log(response.data)
            $('#select-modeltype').html("<option value='' disabled selected>Model Types...</option>");
            $.each(response.data, function (key, value) {
                $('#select-modeltype')
                    .append($("<option></option>")
                    .attr("value", value)
                    .text(value));
            });

            // restore any previously saved session information
            if (response['Type_Name'] !== '') {
                //restore the model type from a previously configured session
                $("#select-modeltype").val(response['Type_Name'])
            }

            if (response['Model_Name'] !== '') {
                // Activate "chosen" for ml model selection drop-down
                getMachineLearningModels()
            }
            $("#select-modeltype").trigger("chosen:updated");
        },
        error: function (response) {
            // todo: research JS error handling framework
            alert("error in getting from server api getModelTypes(), please report to administrators.");
        },
    });
}

function getEvoAlgorithms()
{
    // Calls backend API for evolutionary algorithms
    $.ajax({
        url: "getEvolutionaryAlgorithms/",
        type: "post",
        // example ref: data serialization for server callback
        data: { "mlName": JSON.stringify($("#select-machineLearningModel").val()) },
        cache: 'false',
        dataType: "json",
        async: 'true',
        success: function (response) {
            // debug
            //console.log(response.data)
            $('#select-evolutionaryAlgorithms').html("<option value='' disabled selected>Evolutionary Algorithms...</option>");
            // populate evolutionary algorithms drop-down
            $.each(response.data, function (key, value) {
                $('#select-evolutionaryAlgorithms')
                    .append($("<option></option>")
                    .attr("value", value)
                    .text(value));
            });

            // restore any previously saved session information
            if (response['Evo_Alg_Name'] !== '') {
                //restore the evolutionary algorithm from a previously configured session
                $("#select-evolutionaryAlgorithms").val(response['Evo_Alg_Name'])
            }

            $("#select-evolutionaryAlgorithms").trigger("chosen:updated");
        },
        error: function (response) {
            alert("error getting results from server api getEvolutionaryAlgorithms(), please report to administrators.")
        },
    });
}

function getImplementationMethods()
{
    // Calls backend API for implementation methods
    $.ajax({
        url: "getImplementationMethods/",
        type: "post",
        data: { "type": JSON.stringify($("#select-modeltype").val()) },
        cache: 'false',
        dataType: "json",
        async: 'true',
        success: function (response) {
            // debug
            // console.log(response.data)
            $('#select-implementationMethod').html("<option value='' disabled selected>Implementation Methods...</option>");
            // populate machine learning drop-down selector
            $.each(response.data, function (key, value) {
                $('#select-implementationMethod')
                    .append($("<option></option>")
                    .attr("value", value)
                    .text(value));
            });

            // restore any previously saved session information
            if (response['Impl_Desc'] !== '') {
                //restore the implementation method from a previously configured session
                $("#select-implementationMethod").val(response['Impl_Desc'])
            }

            $("#select-implementationMethod").trigger("chosen:updated");
        },
        error: function (response) {
            alert("error getting results from server api getImplementationMethods(), please report to administrators.")
        },
    });
}

function updateClock() {
    $(".clock").html(moment().format('YYYY-MM-DD HH:mm:ss'));
}

// jQuery Terminal support
//TODO Refactor
function typed(finish_typing) {
    return function (term, message, delay, finish) {
        anim = true;
        var prompt = term.get_prompt();
        var c = 0;
        if (message.length > 0) {
            term.set_prompt('');
            var interval = setInterval(function () {
                term.insert(message[c++]);
                if (c == message.length) {
                    clearInterval(interval);
                    // execute in next interval
                    setTimeout(function () {
                        // swap command with prompt
                        finish_typing(term, message, prompt);
                        anim = false
                        finish && finish();
                    }, delay);
                }
            }, delay);
        }
    };
}

$(function () {
    //jQuery Terminal support functions
    // TODO: refactor
    var anim = false;
    var typed_prompt = typed(function (term, message, prompt) {
        // swap command with prompt
        term.set_command('');
        term.set_prompt(message + ' ');
    });
    var typed_message = typed(function (term, message, prompt) {
        term.set_command('');
        term.echo(message)
        term.set_prompt(prompt);
    });

    // initialize temp data storage in DOM elements
    $("#modelConfigurationConsole").data("dataSetFileList", [])


    // QSAR Console Terminals
    $('#modelConfigurationConsole').terminal(function (command, term) { }, {
        greetings: 'Model Configuration Console',
        name: 'modelConfigurationConsole',
        height: 300,
        width: 550,
        prompt: 'stdout >'
    });

    $('#trainingResults').terminal(function (command, term) {
        if (command !== '') {
            try {
                var result = window.eval(command);
                if (result !== undefined) {
                    term.echo(new String(result));
                }
            } catch (e) {
                term.error(new String(e));
            }
        } else {
            term.echo('');
        }
    },
    {
        greetings: 'Latest Training Results',
        name: 'trainingResults',
        height: 200,
        prompt: 'stdout> '
    });

   // Session Configuration Form fields
    $("#select-modeltype").chosen();
    $("#select-machineLearningModel").chosen();
   // Model type drop-down populated dynamically with database data
    getModelTypes()

    // Populate evolutionary algorithms drop-down with EA implementations list
   $("#select-evolutionaryAlgorithms").chosen();
   getEvoAlgorithms()

   // model type selection triggers a callback to server
   // to populate machineLearningModel drop-down with models associated
   // with the user's selected model type
   $("#select-modeltype").change(function (value) {
       getMachineLearningModels();
   });
   $("#select-implementationMethod").chosen();
   getImplementationMethods()

   // blockui
   //$.blockUI({ message: $('#question'), css: { width: '275px' } });
    // button decorators
   $("input[type=submit]").button()
   $(".loadData").button()
   $(".pullData").button()
   $("#start_qsar_session").button()

   $(".trainingStatusButton").button()
   $(".trainingStatusButton").click(function () {
       if ($('input[id="sessionid"]').val() !== '') {
           // Calls backend API
           dataSetFileList = []
           for (var i = 0; i < $("#modelConfigurationConsole").data("dataSetFileList").length; ++i)
           {
               dataSetFileList.push($("#modelConfigurationConsole").data("dataSetFileList")[i]);
           }

           $.ajax({
               url: "getLatestStatus/",
               type: "post",
               data: { "taskid": $('input[id="taskid"]').val(), "dataSetFileList": JSON.stringify(dataSetFileList) },
               dataType: "json",
               async: 'true',
               success: function (response) {
                   // Show the configuration terminal
                   $("#modelConfigurationConsole").show();
                   var msg = [moment().format('YYYY-MM-DD HH:mm:ss'), response.data].join('  ');
                   $("#modelConfigurationConsole").terminal().echo(msg);
                   $('#taskid').val(response.taskid)
                   if (response.outputfile !== '')
                   {
                       // If output is specified, then build relative file path 
                       //and add reference link to anchor tag
                       outputfilecomponents = response.outputfile.split('\\');
                       relativeFilePath     = outputfilecomponents.slice(-4);
                       relativeFilePath.unshift('..');
                       relativeFilePath = relativeFilePath.join('/');
                       $('.pullData').attr('href', relativeFilePath);
                       // Update result terminal/console window with latest path reference
                       var msg = [moment().format('YYYY-MM-DD HH:mm:ss'), "Latest result file: " + relativeFilePath].join('  ');
                       $("#trainingResults").terminal().echo(msg);
                   }

               },
               error: function (response) {
                   // todo: research JS error handling framework
                   console.log("error in getting from server api startTraining(...), please report to administrators.");
               },
           });
       }
   });

   // Initialize modal dialogs
   $("#model_config_dialog").dialog({ height: "auto", width: "auto", autoOpen: false });
   $("#model_config_dialog").accordion({
       collapsible: true
       , heightStyle: "content"
   });

    //resets clicked attribute
   $(".loadData").click(function () {
       $(".loadData", $(this).parents("form")).removeAttr("clicked");
       $(this).attr("clicked", "true");
   });

   $('#trainingDataFile').fileupload({
       dataType: 'json',
       submit: function (e, data) {
           // Don't allow uploads if a session has not been started
           if ($('input[id="sessionid"]').val() === '') {
               // Show the configuration terminal
               $("#modelConfigurationConsole").show();
               var msg = [moment().format('YYYY-MM-DD HH:mm:ss'), 'Please create a new session before trying to upload a training data file.'].join('  ');
               $("#modelConfigurationConsole").terminal().echo(msg);
               $("#model_config_dialog").dialog("close");
               return false;
           }

           // For now we only accept (well-formatted) CSV files
           if (data.files[0].name.slice(-3) !== 'csv')
           {
               // Show the configuration terminal
               $("#modelConfigurationConsole").show();
               var msg = [moment().format('YYYY-MM-DD HH:mm:ss'), 'Please make sure your data file is a well-formatted CSV'].join('  ');
               $("#modelConfigurationConsole").terminal().echo(msg);
               $("#model_config_dialog").dialog("close");
               return false;
           }
       },
       done: function (e, data) {
           var msg = [moment().format('YYYY-MM-DD HH:mm:ss'), data.result.data].join('  ');
           $("#modelConfigurationConsole").terminal().echo(msg);
           $("#model_config_dialog").dialog("close");
       }
   });

   $("input[type=submit]").click(function () {
       $(".loadData", $(this).parents("form")).removeAttr("clicked");
       $(this).attr("clicked", "false");
   });
    
   // Machine Learning Model configuration form submission logic
   $("#model_config_dialogForm").submit(function (e) {
       e.preventDefault();
       // don't call the server if the load data button
       // was clicked
       if ($(".loadData[clicked=true]").size() > 0) {
           return;
       }
       var serializedData = $("#model_config_dialogForm").serialize();
       var mlType         = $.trim(JSON.stringify($($(".chosen-single span")[0]).text()))
       var mlName         = $.trim(JSON.stringify($("#select-machineLearningModel").val()))
       var eaName         = $.trim(JSON.stringify($("#select-evolutionaryAlgorithms").val()))
       var imName         = $.trim(JSON.stringify($("#select-implementationMethod").val()))
       var isMlTypDefault = (mlType === '"Model Types..."') ? true : false;
       var fieldsEmpty    = (mlType === "" || mlName === "" || eaName === "" || imName === "")
       var fieldsNull     = (mlType === "null" || mlName === "null" || eaName === "null" || imName === "null")
       if (isMlTypDefault || fieldsEmpty || fieldsNull) {
           alert("Please select at least one entry from each field")
       }
       else {
           $.ajax({
               url: "saveSession/",
               type: "post",
               data: { "mlType": mlType, "mlName": mlName, "eaName": eaName, "imName": imName },
               cache: 'false',
               dataType: "json",
               async: 'true',
               success: function (response) {
                   // Get session ID and reroute to dashboard context
                   console.log(response.data.msg)
                   console.log(response.data.sessiondid)
                   appBreadCrumbs       = window.location.pathname.split('/').filter(function (element) { return element !== "" })
                   appRoot              = appBreadCrumbs[0]
                   // Redirect to the new session context
                   window.location.href = [window.location.origin, appRoot, response.data.sessiondid].join('/')
               },
               error: function (response) {
                   alert("error in getting from server")
               },
           });
       }
   });

    //Tabs
    $("#tabs").tabs().addClass("ui-tabs-vertical ui-helper-clearfix");
    $("#tabs li").removeClass("ui-corner-top").addClass("ui-corner-left");

    //Toolbar buttons
    $('.copy-sessionid-button').button({
        text: true
    })

    $("#update_model_config").button({
        text: true
    })

    $("#startTraining").button({
        text: true
    })

    $("#startTraining").click(function () {
        if ($('input[id="sessionid"]').val() !== '') {
            // Calls backend API
            dataSetFileList = []
            for (var i = 0; i < $("#modelConfigurationConsole").data("dataSetFileList").length; ++i) {
                dataSetFileList.push($("#modelConfigurationConsole").data("dataSetFileList")[i]);
            }

            $.ajax({
                url: "startTraining/",
                type: "post",
                data: { "taskid": $('input[id="taskid"]').val(), "dataSetFileList": JSON.stringify(dataSetFileList) },
                dataType: "json",
                async: 'true',
                success: function (response) {
                    // Show the configuration terminal
                    $("#modelConfigurationConsole").show();
                    var msg = [moment().format('YYYY-MM-DD HH:mm:ss'), response.data].join('  ');
                    $("#modelConfigurationConsole").terminal().echo(msg);
                    $('#taskid').val(response.taskid);
                    $("#modelConfigurationConsole").data("dataSetFileList", response.dataSetFileList);
                },
                error: function (response) {
                    // todo: research JS error handling framework
                    console.log("error in getting from server api startTraining(...), please report to administrators.");
                },
            });
        }
    });


    $("#stopTraining").button({
        text: true
    })

    $("#stopTraining").click(function () {
        if ($('input[id="sessionid"]').val() !== '') {
            // Calls backend API
            dataSetFileList = []
            for (var i = 0; i < $("#modelConfigurationConsole").data("dataSetFileList").length; ++i) {
                dataSetFileList.push($("#modelConfigurationConsole").data("dataSetFileList")[i]);
            }

            $.ajax({
                url: "stopTraining/",
                type: "post",
                data: { "taskid": $('input[id="taskid"]').val(), "dataSetFileList": JSON.stringify(dataSetFileList) },
                dataType: "json",
                async: 'true',
                success: function (response) {
                    // Show the configuration terminal
                    $("#modelConfigurationConsole").show();
                    var msg = [moment().format('YYYY-MM-DD HH:mm:ss'), "Task successfully terminated", response.data].join('  ');
                    $("#modelConfigurationConsole").terminal().echo(msg);
                },
                error: function (response) {
                    // todo: research JS error handling framework
                    console.log("error in getting from server api stopTraining( ... ), please report to administrators.");
                },
            });
        }
    });

    $('.help').button({
        label: 'Test',
        icons: { primary: 'ui-icon-custom', secondary: null }
    });

    // Button handlers
    $("#start_qsar_session").click(function () {
        $("#model_config_dialog").accordion({ active: 0 });
        // Activate model configuration tab
        $("#tabs").tabs({ active: 1 });
        // Hide the configuration terminal
        $("#modelConfigurationConsole").hide();
        $("#model_config_dialog").dialog("open");
    });

    $("#update_model_config").click(function () {
        $("#model_config_dialog").dialog("open");
    });

    $(".loadData").click(function () {
        $("#model_config_dialog").accordion({ active: 1 });
    });

    // Zeroclipboard
    ZeroClipboard.config({ swfPath: "/static/qsar/scripts/external/zeroclipboard-2.1.6/ZeroClipboard.swf" });
    var copySessionidButton = new ZeroClipboard($(".copy-sessionid-button"));
    copySessionidButton.on("ready", function (readyEvent) {
        // alert( "ZeroClipboard SWF is ready!" );
        copySessionidButton.on("aftercopy", function (event) {
            // `this` === `client`
            // `event.target` === the element that was clicked
            //event.target.style.display = "none";
            //alert("Copied text to clipboard: " + event.data["text/plain"]);
            var msg = [moment().format('YYYY-MM-DD HH:mm:ss'), 'Session ID copied  to clipboard'].join('  ');
            $("#modelConfigurationConsole").terminal().echo(msg);
        });
    });

    // If this is a session context focus on the model configuration console 
    // under the 'Configure Model' tab
    if ( $('input[id="sessionid"]').val() !== '' ) {
        // Activate model configuration tab
        $("#tabs").tabs({ active: 1 });
        // Show the configuration terminal
        $("#modelConfigurationConsole").show();
        var msg = [moment().format('YYYY-MM-DD HH:mm:ss'), 'Loaded user session id: ', $('input[id="sessionid"]').val()].join('  ');
        $("#modelConfigurationConsole").terminal().echo(msg);
        $(".copy-sessionid-button").attr('data-clipboard-text', msg)
    }

    //Initialize clock
    setInterval('updateClock()', 1000);
});