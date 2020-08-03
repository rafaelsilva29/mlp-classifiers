<template>
  <v-app id="main" :style="{background: $vuetify.theme.themes[theme].background}">
    <v-app-bar
      app
      color="primary"
      dark
    >
      <div class="d-flex align-center">
        <v-img
          alt="Predictions"
          class="shrink mr-2 over"
          contain
          src="./assets/logo.png"
          transition="scale-transition"
          width="50"
          @click="$router.push('/')"
        />
        <h1 class="ml-2 over" @click="$router.push('/')">OML 2020</h1>
      </div>
      <v-spacer></v-spacer>
      <router-link to="/aboutus">
        <v-btn text>
            <span>About Us</span>
            <v-icon class="ml-2">
              mdi-open-in-new
            </v-icon>
        </v-btn>
      </router-link>
    </v-app-bar>
    <v-content>
      <router-view/>
    </v-content>
    <div id="footer">
      <v-footer
        :style="{background: $vuetify.theme.themes[theme].background}"
        class="font-weight-medium"
      >
        <v-col
          class="text-center pb-1"
          cols="12"
        >
          <strong>OML {{ new Date().getFullYear() }} - Grupo A </strong>
        </v-col>
         <v-col
          class="text-center pa-0 ma-0"
          cols="12"
        >
          <p>MLFA - Machine Learning Fundamentos e Aplicações</p>
        </v-col>
        <div class="pa-2"></div>
      </v-footer>
    </div>
  </v-app>
</template>

<script>
export default {
  name: 'App',
  computed:{
    theme(){
      return (this.$vuetify.theme.dark) ? 'dark' : 'light'
    }
  }
};
</script>

<style lang="scss">
#footer {
  display: table-row;
}
.over {
  cursor: pointer;
}
p {
  font-size: 14px;
}
</style>
