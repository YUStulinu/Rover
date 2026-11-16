// WebGL2 renderer of colored lines and points.

import type { Geometry } from "./scene.js";
import type { Mat4 } from "./math.js";

const VS = `#version 300 es
in vec3 pos;
in vec3 col;
uniform mat4 mvp;
uniform float size;
out vec3 v_col;
void main() {
  gl_Position = mvp * vec4(pos, 1.0);
  gl_PointSize = size;
  v_col = col;
}`;

const FS = `#version 300 es
precision mediump float;
in vec3 v_col;
uniform float round_points;
out vec4 color;
void main() {
  if (round_points > 0.5 && length(gl_PointCoord - 0.5) > 0.5) discard;
  color = vec4(v_col, 1.0);
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
  return s;
}

export class Renderer {
  private gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private lineBuf: WebGLBuffer;
  private pointBuf: WebGLBuffer;
  private nLines = 0;
  private nPoints = 0;
  private cursorStart = 0; // the trailing points are the time cursors (drawn larger)

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: true });
    if (!gl) throw new Error("WebGL2 is not available");
    this.gl = gl;
    const p = gl.createProgram()!;
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link");
    this.prog = p;
    this.lineBuf = gl.createBuffer()!;
    this.pointBuf = gl.createBuffer()!;
    gl.enable(gl.DEPTH_TEST);
  }

  upload(g: Geometry) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
    gl.bufferData(gl.ARRAY_BUFFER, g.lines, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointBuf);
    gl.bufferData(gl.ARRAY_BUFFER, g.points, gl.DYNAMIC_DRAW);
    this.nLines = g.lines.length / 6;
    this.nPoints = g.points.length / 6;
    this.cursorStart = this.nPoints - g.cursors;
  }

  private bind(buf: WebGLBuffer) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    const pos = gl.getAttribLocation(this.prog, "pos");
    const col = gl.getAttribLocation(this.prog, "col");
    gl.enableVertexAttribArray(pos);
    gl.vertexAttribPointer(pos, 3, gl.FLOAT, false, 24, 0);
    gl.enableVertexAttribArray(col);
    gl.vertexAttribPointer(col, 3, gl.FLOAT, false, 24, 12);
  }

  draw(mvp: Mat4, background: [number, number, number], pointSize = 2) {
    const gl = this.gl;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(this.canvas.clientWidth * dpr), h = Math.round(this.canvas.clientHeight * dpr);
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    gl.viewport(0, 0, w, h);
    gl.clearColor(background[0], background[1], background[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.prog, "mvp"), false, mvp);
    const size = gl.getUniformLocation(this.prog, "size");
    const round = gl.getUniformLocation(this.prog, "round_points");
    if (this.nLines) {
      this.bind(this.lineBuf);
      gl.uniform1f(round, 0);
      gl.drawArrays(gl.LINES, 0, this.nLines);
    }
    if (this.nPoints) {
      this.bind(this.pointBuf);
      gl.uniform1f(size, pointSize * dpr);
      gl.uniform1f(round, 0);
      if (this.cursorStart > 0) gl.drawArrays(gl.POINTS, 0, this.cursorStart);
      if (this.cursorStart < this.nPoints) {
        gl.uniform1f(size, 10 * dpr);
        gl.uniform1f(round, 1);
        gl.drawArrays(gl.POINTS, this.cursorStart, this.nPoints - this.cursorStart);
      }
    }
  }
}
