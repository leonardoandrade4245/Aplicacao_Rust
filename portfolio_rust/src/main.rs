/// Módulo responsável pela análise de séries temporais utilizando regressão linear simples.

/// Estrutura que armazena os coeficientes calculados pela regressão linear.
#[derive(Debug)]
pub struct RegressaoLinear {
    pub intercepto: f64,
    pub inclinacao: f64,
}

impl RegressaoLinear {
    /// Realiza o ajuste dos coeficientes (inclinação e intercepto) 
    /// utilizando o método dos mínimos quadrados.
    /// `periodos_x` representa os períodos (ex.: 0, 1, 2, ...).
    /// `valores_y` representa os valores correspondentes da série.
    pub fn ajustar(periodos_x: &[f64], valores_y: &[f64]) -> Result<Self, String> {
        if periodos_x.len() != valores_y.len() || periodos_x.is_empty() {
            return Err("Os vetores devem ter o mesmo tamanho e não podem ser vazios.".to_string());
        }

        let quantidade = periodos_x.len() as f64;

        let soma_x: f64 = periodos_x.iter().sum();
        let soma_y: f64 = valores_y.iter().sum();

        let soma_x_quadrado: f64 = periodos_x.iter().map(|v| v * v).sum();
        let soma_xy: f64 = periodos_x.iter().zip(valores_y.iter()).map(|(xi, yi)| xi * yi).sum();

        let denominador = quantidade * soma_x_quadrado - soma_x * soma_x;
        if denominador == 0.0 {
            return Err("Divisão por zero ao calcular os coeficientes.".to_string());
        }

        let inclinacao = (quantidade * soma_xy - soma_x * soma_y) / denominador;
        let intercepto = (soma_y - inclinacao * soma_x) / quantidade;

        Ok(RegressaoLinear {
            intercepto,
            inclinacao,
        })
    }

    /// Realiza a previsão de um valor futuro com base no modelo ajustado.
    pub fn prever(&self, periodo_x: f64) -> f64 {
        self.intercepto + self.inclinacao * periodo_x
    }

    /// Calcula o coeficiente de determinação R²,
    /// indicando o quão bem a linha ajustada representa os dados observados.
    pub fn r2(&self, periodos_x: &[f64], valores_y: &[f64]) -> f64 {
        let media_y: f64 = valores_y.iter().sum::<f64>() / valores_y.len() as f64;

        let soma_total: f64 = valores_y.iter().map(|yi| (yi - media_y).powi(2)).sum();
        let soma_residual: f64 = periodos_x.iter().zip(valores_y.iter())
            .map(|(xi, yi)| {
                let y_estimado = self.prever(*xi);
                (yi - y_estimado).powi(2)
            })
            .sum();

        1.0 - (soma_residual / soma_total)
    }

    /// Calcula o Erro Quadrático Médio (MSE) do modelo ajustado.
    pub fn mse(&self, periodos_x: &[f64], valores_y: &[f64]) -> f64 {
        let quantidade = valores_y.len() as f64;
        let erro_total: f64 = periodos_x.iter().zip(valores_y.iter())
            .map(|(xi, yi)| {
                let y_estimado = self.prever(*xi);
                (yi - y_estimado).powi(2)
            })
            .sum();
        erro_total / quantidade
    }
}

fn main() {
    // Exemplo de uso da regressão linear

    // Dados da série temporal: valores y e seus respectivos períodos x (0,1,2,3,4)
    let valores_y = [2.0, 3.0, 5.0, 7.0, 11.0];
    let periodos_x: Vec<f64> = (0..valores_y.len()).map(|v| v as f64).collect();

    // Ajustando o modelo de regressão linear
    let modelo = RegressaoLinear::ajustar(&periodos_x, &valores_y)
        .expect("Erro no cálculo da regressão");

    println!("Modelo ajustado: {:?}", modelo);
    println!("Coeficiente de determinação R²: {:.4}", modelo.r2(&periodos_x, &valores_y));
    println!("Erro quadrático médio (MSE): {:.4}", modelo.mse(&periodos_x, &valores_y));

    // Fazendo previsões para os períodos futuros 5, 6 e 7
    for periodo in 5..8 {
        let previsao = modelo.prever(periodo as f64);
        println!("Previsão para t = {}: {:.4}", periodo, previsao);
    }
}


/// Testes Unitários
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn testa_calculo_coeficientes() {
        let valores_y = [2.0, 4.0, 6.0, 8.0];
        let periodos_x: Vec<f64> = (0..valores_y.len()).map(|v| v as f64).collect();

        let modelo = RegressaoLinear::ajustar(&periodos_x, &valores_y).unwrap();

        assert!((modelo.inclinacao - 2.0).abs() < 1e-6);
        assert!((modelo.intercepto - 2.0).abs() < 1e-6);
    }

    #[test]
    fn testa_previsao() {
        let valores_y = [1.0, 3.0, 5.0, 7.0];
        let periodos_x: Vec<f64> = (0..valores_y.len()).map(|v| v as f64).collect();

        let modelo = RegressaoLinear::ajustar(&periodos_x, &valores_y).unwrap();

        let previsao = modelo.prever(4.0);
        assert!((previsao - 9.0).abs() < 1e-6);
    }

    #[test]
    fn testa_r2_perfeito() {
        let valores_y = [2.0, 4.0, 6.0, 8.0];
        let periodos_x: Vec<f64> = (0..valores_y.len()).map(|v| v as f64).collect();

        let modelo = RegressaoLinear::ajustar(&periodos_x, &valores_y).unwrap();

        let r2 = modelo.r2(&periodos_x, &valores_y);
        assert!((r2 - 1.0).abs() < 1e-6);
    }
}
