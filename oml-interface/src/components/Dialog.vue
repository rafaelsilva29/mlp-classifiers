<template>
    <v-container>
        <v-layout row justify-center>
            <v-dialog v-model="dialog" fullscreen hide-overlay transition="dialog-bottom-transition">
                <v-card class="mx-auto">
                    <v-toolbar dark color="primary">
                        <v-btn icon dark @click.prevent="closeDialog">
                            <v-icon>mdi-close</v-icon>
                        </v-btn>
                        <v-toolbar-title>More info about the classifier</v-toolbar-title>
                    </v-toolbar>
                    <v-card-title class="mb-3" style="text-transform: capitalize;">{{ data.perceptron }} - {{ data.classifier }} - {{ data.dataset }}</v-card-title>
                    <v-card-subtitle class="pb-0">
                        <v-divider class='mt-4 mb-2'></v-divider>
                        <h3>Evaluation</h3>
                        <ul>
                            <li>Score: {{ data.score }} - [0-1]</li>
                            <li>Time: {{ data.time }} s</li>
                        </ul>
                    </v-card-subtitle>
                    <v-card-text class="text--primary align-center">
                        <v-divider class='mt-4 mb-4'></v-divider>
                        <h2 style="text-align: center;">Plots</h2>
                        <h4>Errors</h4>
                        <v-row>
                            <v-col cols="3" md="3" v-for="item in data.errors" :key="item">
                                <v-img class="pa-0" transition="scale-transition" style="margin: 0 auto;" max-width="1000" v-bind:src="'data:image/png;base64,'+item" />
                            </v-col>
                        </v-row>
                        <v-divider class='mt-4 mb-2'></v-divider>
                        <h4>Confusion Matrix</h4>
                        <v-row>
                            <v-col cols="12" md="12">
                                <v-img class="pa-0" transition="scale-transition" style="margin: 0 auto;" max-width="700" v-bind:src="'data:image/png;base64,'+data.confusion_matrix_image" />
                            </v-col>
                        </v-row>
                        <div v-if="data.predictions_image">
                            <v-divider class='mt-4 mb-2'></v-divider>
                            <h4>Predictions Image</h4>
                            <v-row>
                                <v-col cols="12" md="12">
                                    <v-img class="pa-0" transition="scale-transition" style="margin: 0 auto;" max-width="700" v-bind:src="'data:image/png;base64,'+data.predictions_image" />
                                </v-col>
                            </v-row>
                        </div>
                        <v-divider class='mt-4 mb-2'></v-divider>
                    </v-card-text>
                </v-card>
            </v-dialog>
        </v-layout>
    </v-container>
</template>

<script>
  export default {
    name: 'Dialog',
    props: {
        dialog: Boolean,
        data: Object
    },
    methods: {
        closeDialog(){
            this.$emit('closeDialog', 'close')
        }
    },
  }
</script>
