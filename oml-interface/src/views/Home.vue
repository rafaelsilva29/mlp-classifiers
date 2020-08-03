<template>
  <div id="home">
    <v-container fluid>
      <v-row align-content="center">
        <v-col cols="6" md="4">
          <v-card
            class="mx-auto"
            max-width="400"
          >
            <v-img
              class="white--text align-end"
              height="200px"
              src="../assets/docks.png"
            >
              <v-card-title>Menu</v-card-title>
            </v-img>
            <v-card-text class="text--primary align-center">
              <v-form ref='form' v-model='valid' lazy-validation>
                <h2>Datasets</h2>
                <v-radio-group v-model="dataset" row :rules=[rules.required] required>
                  <v-radio label="Digits" value="digits"></v-radio>
                  <v-radio label="Mnist" value="mnist"></v-radio>
                </v-radio-group>
                <v-divider class='mb-4 mt-1'></v-divider>
                <h2>Perceptron type</h2>
                <v-combobox
                  v-model="perceptron"
                  :items="itemsPerceptron"
                  :rules="[rules.requiredMin]"
                  placeholder="Choice min one perceptron"
                  small-chips
                ></v-combobox>
                <div v-if="polynominal">
                  <v-divider class='mb-4 mt-1'></v-divider>
                  <h3>Degree</h3>
                  <v-text-field
                    v-model="degree"
                    :rules="[rules.required]"
                    placeholder="Degree polynominal"
                    type="number"
                  ></v-text-field>
                </div>
                <v-divider class='mb-4 mt-1'></v-divider>
                <v-row>
                  <v-col cols="6" md=6>
                    <h2>Max iterations</h2>
                      <v-text-field
                        v-model="iterations"
                        :rules="[rules.required]"
                        placeholder="Max iterations"
                        type="number"
                      ></v-text-field>
                  </v-col>
                  <v-col cols="6" md=6 v-if="primal">
                    <h2>Learning Rate</h2>
                    <v-text-field
                      v-model="learningrate"
                      :rules="[rules.required]"
                      placeholder="Learning Rate"
                      type="number"
                    ></v-text-field>
                  </v-col>
                </v-row>
                <v-divider class='mb-4 mt-1'></v-divider>
                <h2>Classes</h2>
                <v-combobox
                  v-model="classes"
                  :items="itemsClasses"
                  :rules="[rules.requiredList, rules.requiredListMax]"
                  multiple
                  placeholder="Choice min two classes"
                  small-chips
                ></v-combobox>
                <v-divider class='mb-4 mt-1'></v-divider>
                <h2>Classifier</h2>
                <v-combobox
                  v-model="classfiers"
                  :items="itemsClassifiers"
                  :rules="[rules.requiredMin]"
                  multiple
                  placeholder="Choice min one classifier"
                  small-chips
                ></v-combobox>
              </v-form>
            </v-card-text>

            <v-card-actions>
              <v-btn
                color="orange"
                text
                @click.prevent="validate"
                :disabled="disable"
              >
                Make predictions
              </v-btn>

              <v-btn
                color="orange"
                text
                @click="$router.go(0)"
              >
                Clean
              </v-btn>
            </v-card-actions>
          </v-card>
        </v-col>
        <v-col cols="6" md="8">
          <div v-if="status == 1" class="d-flex justify-center center">
            <v-progress-circular
              :rotate="-90"
              :size="350"
              :width="10"
              :value="value"
              color="primary"
            >
              {{ value }}
            </v-progress-circular>
          </div>
          <div v-else-if="status == 2">
            <v-card
              class="mx-auto"
            >
              <v-card-title class="mb-3" style="text-transform: capitalize;">Predictions of {{ perceptron }} in {{ dataset }}</v-card-title>
              <v-card-subtitle class="pb-0">
                <h3>Classes</h3>
                <ul>
                  <li v-for="item in classes" :key="item">
                    {{ item }}
                  </li>
                </ul>
              </v-card-subtitle>
              <v-card-text class="text--primary"> 
                <v-divider class='mt-4 mb-2'></v-divider>
                <v-data-table
                  class="pl-4 pr-4"
                  :headers="headers"
                  :items="resume"
                  @click:row="clickRow"
                ></v-data-table>
                <v-row>
                  <v-col cols="12" md="12" v-for="item in images" :key="item">
                    <v-img class="pa-0" style="margin: 0 auto;"  transition="scale-transition" max-width="500" v-if="item != ''" v-bind:src="'data:image/png;base64,'+item" />
                  </v-col>
                </v-row>
              </v-card-text>
            </v-card>
          </div>
        </v-col>
      </v-row>
      <Dialog :dialog="dialog" :data="data" v-on:closeDialog="closeDialog" v-if="dialog"/>
    </v-container>
  </div>
</template>

<script>
import Dialog from '@/components/Dialog.vue'

export default {
  name: 'Home',
  components: {
    Dialog,
  },
  data() {
    return {
      dialog: false,
      data: {},
      disable: false,
      value: 0,
      status: 0,
      valid: true,
      dataset: '',
      perceptron: '',
      degree: 2,
      iterations: 10,
      learningrate: 0.1,
      classes: [],
      classfiers: [],
      itemsClasses: [0,1,2,3,4,5,6,7,8,9],
      itemsPerceptron: ['primal','dual','rbf','polynomial'],
      itemsClassifiers: ['ova','ovo','ecoc'],
      rules: {},
      explore_image: '',
      confusion_matrix: '',
      predictions_image: '',
      search: '',
      headers: [
        { text: 'Classifier', value: 'classifier' },
        { text: 'Score - [0-1]', value: 'score' },
        { text: 'Time - (s)', value: 'time' },
      ],
      resume: [],
      response_all: [],
      images: []
    }
  },
  created() {
    this.rules = {
      required: (value) => !!value || 'Required',
      requiredMin: (v) => (v && v.length >= 1) || 'Choice at least on option',
      requiredList: (v) => (v && v.length >= 2) || 'Choice min two classes',
      requiredListMax: (v) => (v && v.length <= 5) || 'Max five classes',
    };
  },
  methods: {
    clickRow(value) {
      this.data = value
      this.dialog = true;
    },
    closeDialog(){
      this.dialog = false
    },
    async validate() {
      if (this.$refs.form.validate()) { 
        this.status = 1
        this.disable = true
        const headers = {
          'Content-Type': 'application/json',
        }
        let data = {
          dataset: this.dataset,
          perceptron_type: this.perceptron,
          classes: this.classes,
          max_iterations: this.iterations,
        }
        if (this.perceptron == 'primal') {
          data.learning_rate = this.learningrate
        } else if (this.perceptron != 'primal') {
          if(this.perceptron == 'dual') {
            data.perceptron_type = this.perceptron
            data.kernel = 'none'
          } else if (this.perceptron == 'polynomial') {
            data.perceptron_type = 'dual'
            data.kernel = 'polynomial'
            data.degree = this.degree
          } else {
            data.perceptron_type = 'dual'
            data.kernel = this.perceptron
          }
        }
        try {
          let response_temp = [];
          for(let i=0; i<this.classfiers.length; i++) {
            this.value += 10
            let classifier = this.classfiers[i]
            data.classifier = classifier
            const response = await this.axios.post('/predict', data, { headers })
            response_temp.push(response.data)
            let temp = {
              perceptron: this.perceptron,
              dataset: this.dataset,
              classifier: response.data.title,
              score: response.data.score,
              time: response.data.time,
              confusion_matrix_image: this.fixImage(response.data.confusion_matrix_image),
              errors: this.fixImageList(response.data.errors),
              iterations: response.data.iterations
            }
            if (this.dataset == 'digits') {
              temp.predictions_image = this.fixImage(response.data.predictions_image)
            }
            this.resume.push(temp)
            this.value += 25
          }
          let aux = 100
          this.value += (aux-this.value)
          setTimeout(() => {
            this.response_all = response_temp
            this.images.push(this.fixImage(response_temp[0].image_explore))
            this.status = 2;
            this.value = 0
          }, 2000);
        } catch (error) {
          this.disable = false
        }
      }
    },
    fixImage(img) {
      let new_img = img.substr(2);
      new_img = new_img.substring(0, new_img.length - 1);
      return new_img
    },
    fixImageList(imgList) {
      let img_temp = []
      for(let i=0; i<imgList.length; i++) {
        let new_img = imgList[i].substr(2);
        new_img = new_img.substring(0, new_img.length - 1);
        img_temp.push(new_img)
      }
      return img_temp
    },
  },
  computed: {
    polynominal() {
      let flag = false;
      if (this.perceptron == 'polynomial') {
        flag = true;
      }
      return flag; 
    },
    primal() {
      let flag = false;
      if (this.perceptron == 'primal') {
        flag = true;
      }
      return flag; 
    }
  },
}
</script>

<style lang="scss">
.center {
  margin-top: 225px;
}
#home
  tr td {
    cursor: pointer !important;
  }
</style>